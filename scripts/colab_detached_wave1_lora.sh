#!/usr/bin/env bash
# Anti-Idle-Burn detached Colab runner for Wave 1 LoRA batches (runs inside WSL Ubuntu).
# Usage: bash colab_detached_wave1_lora.sh <batch_name> <methods_csv_or_space> <protein1> [protein2 ...]
# Example: bash colab_detached_wave1_lora.sh batch1_oxda "lora_llr lora_linear_head" OXDA_RHOTO_Vanella_2023_activity
set -u

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <batch_name> <methods_space_sep> <protein1> [protein2 ...]"
  exit 2
fi

BATCH_NAME="$1"
METHODS="$2"
shift 2
PROTEINS="$*"

COLAB="/home/eva/.local/bin/colab"
SESSION="w1_${BATCH_NAME}"
GPU_TYPE="L4"
MAX_WALL_SEC=2400   # 40-minute hard wall-clock limit per batch
POLL_SEC=30
REPO_LOCAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd) || exit 1
OUT_LOCAL_DIR="${WAVE1_OUT_DIR:-$REPO_LOCAL/outputs/wave1_lora}"
mkdir -p "$OUT_LOCAL_DIR/records" || exit 1
exec 9>"$REPO_LOCAL/outputs/.wave1-lora.lock"
flock -n 9 || { echo "[Error] Another Wave 1 batch is active."; exit 1; }

ROOT_REMOTE=/content/work/wave1_lora
START_TS=$(date +%s)

cleanup() {
  echo "=== [Cleanup] Unconditionally stopping session '$SESSION' & sweeping any orphans ==="
  timeout 30 "$COLAB" --auth=oauth2 stop -s "$SESSION" || true
  timeout 30 /home/eva/.local/share/uv/tools/google-colab-cli/bin/python - <<'PYEOF' || true
from colab_cli.auth import AuthProvider
from colab_cli.common import state
state.auth_provider = AuthProvider.OAUTH2
for a in state.client.list_assignments():
    print(f"[Sweep] Unassigning server assignment: {a.endpoint} ({a.accelerator})")
    state.client.unassign(a.endpoint)
PYEOF
  sleep 2
  echo "=== [Cleanup] Post-flight session & CU verification ==="
  timeout 30 "$COLAB" --auth=oauth2 sessions || true
  local usage
  if ! usage=$(timeout 30 "$COLAB" --auth=oauth2 usage) || [[ "$usage" != *"Active assignments: 0"* || "$usage" != *"Usage rate: 0.00/hr"* ]]; then
    echo "[Error] GPU release is unverified: $usage"
    exit 1
  fi
  echo "$usage"
  if [ -n "${TMP_DIR:-}" ]; then
    rm -r -- "$TMP_DIR"
  fi
}

echo "=== [0/5] Pre-flight check (no orphan sessions & starting CU) ==="
timeout 30 "$COLAB" --auth=oauth2 sessions || exit 1
USAGE=$(timeout 30 "$COLAB" --auth=oauth2 usage) || exit 1
echo "$USAGE"
if [[ "$USAGE" != *"Active assignments: 0"* ]]; then
  echo "[Error] Existing Colab assignment detected."
  exit 1
fi
trap cleanup EXIT
trap 'exit 130' INT TERM

TMP_DIR=$(mktemp -d "$REPO_LOCAL/outputs/.wave1-tmp.XXXXXX") || exit 1
CODE_COMMIT=$(git -C "$REPO_LOCAL" rev-parse HEAD) || exit 1
tar -czf "$TMP_DIR/source.tar.gz" -C "$REPO_LOCAL" --exclude='__pycache__' \
  configs src/esm2_ood pyproject.toml README.md \
  scripts/run_wave1_lora_colab.py scripts/sync_wave1_lora_archive.py || exit 1
SOURCE_SUM=$(sha256sum "$TMP_DIR/source.tar.gz") || exit 1
PYTHONPATH="$REPO_LOCAL/src" python3 -B "$REPO_LOCAL/scripts/sync_wave1_lora_archive.py" \
  pack "$OUT_LOCAL_DIR/records" "$TMP_DIR/checkpoints.tar.gz" || exit 1

echo "=== [1/5] Provisioning $GPU_TYPE session '$SESSION' (timeout=180s) ==="
timeout 180 "$COLAB" --auth=oauth2 new -s "$SESSION" --gpu "$GPU_TYPE" || exit 1
sleep 4

echo "=== [2/5] Warm-check GPU & kernel responsiveness (max 3 attempts) ==="
WARM_OK=0
for attempt in 1 2 3; do
  if WARM_OUT=$(echo "import subprocess; print('GPU_READY:', subprocess.check_output(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip())" | timeout 30 "$COLAB" --auth=oauth2 exec -s "$SESSION" --timeout 20) && [[ "$WARM_OUT" == *"GPU_READY: NVIDIA L4"* ]]; then
    WARM_OK=1
    break
  fi
  echo "[Warn] Warm-check attempt $attempt failed; retrying in 5s..."
  sleep 5
done
if [ "$WARM_OK" -ne 1 ]; then
  echo "[Error] VM warm-check failed 3 times. Aborting before wasting CU."
  exit 1
fi

echo "=== [3/5] Launching Wave 1 LoRA batch '$BATCH_NAME' detached (<5s exec) ==="
timeout 90 "$COLAB" --auth=oauth2 upload -s "$SESSION" "$TMP_DIR/source.tar.gz" /content/wave1_source.tar.gz || exit 1
timeout 90 "$COLAB" --auth=oauth2 upload -s "$SESSION" "$TMP_DIR/checkpoints.tar.gz" /content/wave1_checkpoints.tar.gz || exit 1
cat << PYEOF > "$TMP_DIR/launch.py"
import subprocess, pathlib

work = pathlib.Path("/content/work/wave1_lora")
work.mkdir(parents=True, exist_ok=True)
started = work / "STARTED.flag"
if started.exists():
    print("LAUNCHED_DETACHED_OK", flush=True)
    raise SystemExit(0)

runner_sh = work / "run_bg.sh"
runner_sh.write_text("""#!/usr/bin/env bash
set -euo pipefail
exec > /content/work/wave1_lora/job.log 2>&1
trap 'rc=\$?; if [ "\$rc" -ne 0 ]; then printf "%s\\n" "\$rc" > "$ROOT_REMOTE/FAIL.tmp"; mv "$ROOT_REMOTE/FAIL.tmp" "$ROOT_REMOTE/FAIL.flag"; fi' EXIT
echo "[START] BATCH=${BATCH_NAME}"
mkdir -p "$ROOT_REMOTE/outputs/records" "$ROOT_REMOTE/source"
echo "${SOURCE_SUM%% *}  /content/wave1_source.tar.gz" | sha256sum -c -
tar -xzf /content/wave1_source.tar.gz -C "$ROOT_REMOTE/source"
tar -xzf /content/wave1_checkpoints.tar.gz -C "$ROOT_REMOTE/outputs/records"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip uninstall -y -q torchao || true
pip install -q transformers peft accelerate pyarrow scipy
pip install -q --no-deps -e "$ROOT_REMOTE/source"
python3 -u "$ROOT_REMOTE/source/scripts/run_wave1_lora_colab.py" \\
  --git-commit ${CODE_COMMIT} \\
  --work-dir /content/work/wave1_lora \\
  --out-dir /content/work/wave1_lora/outputs \\
  --methods ${METHODS} \\
  --proteins ${PROTEINS}
echo OK > "$ROOT_REMOTE/DONE.tmp"
mv "$ROOT_REMOTE/DONE.tmp" "$ROOT_REMOTE/DONE.flag"
""")
runner_sh.chmod(0o755)
started.touch(exist_ok=False)
try:
    subprocess.Popen(
        ["nohup", "timeout", "${MAX_WALL_SEC}", "bash", str(runner_sh)],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, start_new_session=True,
    )
except OSError:
    started.unlink()
    raise
print("LAUNCHED_DETACHED_OK", flush=True)
PYEOF

LAUNCH_OK=0
for attempt in 1 2 3; do
  if LAUNCH_OUT=$(timeout 30 "$COLAB" --auth=oauth2 exec -s "$SESSION" --timeout 20 -f "$TMP_DIR/launch.py") && [[ "$LAUNCH_OUT" == *"LAUNCHED_DETACHED_OK"* ]]; then
    LAUNCH_OK=1
    break
  fi
  echo "[Warn] Launch attempt $attempt failed; retrying..."
  sleep 4
done
if [ "$LAUNCH_OK" -ne 1 ]; then
  echo "[Error] Failed to launch detached job. Stopping VM."
  exit 1
fi

echo "=== [4/5] Polling detached job (hard limit=${MAX_WALL_SEC}s) ==="
CONSEC_FAIL=0
STATUS="TIMEOUT"

cat << 'PYEOF' > "$TMP_DIR/poll.py"
import pathlib
w = pathlib.Path("/content/work/wave1_lora")
if (w / "DONE.flag").exists():
    print("STATE:DONE")
elif (w / "FAIL.flag").exists():
    print("STATE:FAIL:" + (w / "FAIL.flag").read_text().strip())
else:
    print("STATE:RUNNING")
rec_dir = w / "outputs" / "records"
n_rec = len(list(rec_dir.glob("*.json"))) if rec_dir.exists() else 0
print(f"RECORDS:{n_rec}")
log = w / "job.log"
if log.exists():
    lines = log.read_text(errors="replace").splitlines()
    for ln in lines[-5:]:
        print("LOG>", ln)
PYEOF

while true; do
  NOW_TS=$(date +%s)
  ELAPSED=$((NOW_TS - START_TS))
  if [ "$ELAPSED" -ge "$MAX_WALL_SEC" ]; then
    echo "[Watchdog] Hit hard wall-clock cap (${ELAPSED}s >= ${MAX_WALL_SEC}s). Stopping."
    STATUS="WALL_CAP"
    break
  fi

  POLL_OUT=""
  if POLL_OUT=$(timeout 30 "$COLAB" --auth=oauth2 exec -s "$SESSION" --timeout 20 -f "$TMP_DIR/poll.py" 2>&1) && [[ "$POLL_OUT" == *"STATE:"* ]]; then
    CONSEC_FAIL=0
    echo "--- Poll @ ${ELAPSED}s ---"
    echo "$POLL_OUT"
    if echo "$POLL_OUT" | grep -q "STATE:DONE"; then
      STATUS="DONE"
      break
    elif echo "$POLL_OUT" | grep -q "STATE:FAIL"; then
      STATUS="FAIL"
      break
    fi
  else
    CONSEC_FAIL=$((CONSEC_FAIL + 1))
    echo "[Warn] Poll failed ($CONSEC_FAIL/4) @ ${ELAPSED}s"
    if [ "$CONSEC_FAIL" -ge 4 ]; then
      echo "[Watchdog] 4 consecutive poll failures. Stopping VM to avoid idle burn."
      STATUS="POLL_ABORT"
      break
    fi
  fi
  sleep "$POLL_SEC"
done

echo "=== [5/5] Retrieving artifacts (STATUS=$STATUS) ==="
timeout 90 "$COLAB" --auth=oauth2 download -s "$SESSION" "$ROOT_REMOTE/job.log" "$TMP_DIR/job.log" &&
  cp -n "$TMP_DIR/job.log" "$OUT_LOCAL_DIR/${BATCH_NAME}_job.log"
if echo "import subprocess; subprocess.run(['python3', '-B', '$ROOT_REMOTE/source/scripts/sync_wave1_lora_archive.py', 'pack', '$ROOT_REMOTE/outputs/records', '/content/wave1_lora_records.tar.gz'], check=True)" |
  timeout 30 "$COLAB" --auth=oauth2 exec -s "$SESSION" --timeout 20 &&
  timeout 90 "$COLAB" --auth=oauth2 download -s "$SESSION" /content/wave1_lora_records.tar.gz "$TMP_DIR/records.tar.gz"; then
  mkdir -p "$TMP_DIR/records"
  tar -xzf "$TMP_DIR/records.tar.gz" -C "$TMP_DIR/records" || exit 1
  PYTHONPATH="$REPO_LOCAL/src" python3 -B "$REPO_LOCAL/scripts/sync_wave1_lora_archive.py" \
    import "$TMP_DIR/records" "$OUT_LOCAL_DIR/records" || exit 1
else
  exit 1
fi
if [ "$STATUS" != "DONE" ]; then
  echo "[Warn] Job ended with STATUS=$STATUS. Check $OUT_LOCAL_DIR/${BATCH_NAME}_job.log"
  exit 1
fi
