#!/usr/bin/env bash
# Anti-Idle-Burn detached Colab runner for Wave 0B (runs inside WSL Ubuntu).
set -u

COLAB="/home/eva/.local/bin/colab"
SESSION="wave0b"
GPU_TYPE="L4"
MAX_WALL_SEC=720   # 12-minute hard wall-clock limit
POLL_SEC=25
OUT_LOCAL_DIR="/mnt/d/Antigravity/projects/esm2-fitness-ood/outputs/wave0b"
mkdir -p "$OUT_LOCAL_DIR"

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
  timeout 30 "$COLAB" --auth=oauth2 usage || true
}
trap cleanup EXIT

echo "=== [0/5] Pre-flight check (no orphan sessions & starting CU) ==="
timeout 30 "$COLAB" --auth=oauth2 sessions
timeout 30 "$COLAB" --auth=oauth2 usage

echo "=== [1/5] Provisioning $GPU_TYPE session '$SESSION' (timeout=180s) ==="
timeout 180 "$COLAB" --auth=oauth2 new -s "$SESSION" --gpu "$GPU_TYPE"
sleep 4

echo "=== [2/5] Warm-check GPU & kernel responsiveness (max 3 attempts) ==="
WARM_OK=0
for attempt in 1 2 3; do
  if echo "import torch; print('GPU_READY:', torch.cuda.get_device_name(0))" | timeout 35 "$COLAB" --auth=oauth2 exec -s "$SESSION" --timeout 25; then
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

echo "=== [3/5] Launching Wave 0B detached in background (<5s exec) ==="
cat << 'PYEOF' > /tmp/launch_wave0b.py
import subprocess, pathlib

work = pathlib.Path("/content/work/wave0b")
work.mkdir(parents=True, exist_ok=True)
for f in ("DONE.flag", "FAIL.flag"):
    (work / f).unlink(missing_ok=True)

runner_sh = work / "run_bg.sh"
runner_sh.write_text("""#!/usr/bin/env bash
set -uo pipefail
exec > /content/work/wave0b/job.log 2>&1
echo "[START] $(date -u)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
pip install -q transformers peft accelerate pyarrow scipy
if [ ! -d /content/esm2-fitness-ood ]; then
  git clone https://github.com/wl2116641973-dev/esm2-fitness-ood.git /content/esm2-fitness-ood
else
  git -C /content/esm2-fitness-ood pull --ff-only
fi
pip install -q --no-deps -e /content/esm2-fitness-ood
python3 -u /content/esm2-fitness-ood/scripts/run_wave0b_colab.py
RC=$?
if [ "$RC" -eq 0 ]; then
  tar -czf /content/work/wave0b_outputs.tar.gz -C /content/work/wave0b/outputs .
  echo "OK" > /content/work/wave0b/DONE.flag
else
  echo "$RC" > /content/work/wave0b/FAIL.flag
fi
echo "[END] $(date -u) RC=$RC"
""")
runner_sh.chmod(0o755)
subprocess.Popen(["nohup", "bash", str(runner_sh)], start_new_session=True)
print("LAUNCHED_DETACHED_OK", flush=True)
PYEOF

LAUNCH_OK=0
for attempt in 1 2 3; do
  if timeout 40 "$COLAB" --auth=oauth2 exec -s "$SESSION" --timeout 30 -f /tmp/launch_wave0b.py; then
    LAUNCH_OK=1
    break
  fi
  echo "[Warn] Launch attempt $attempt failed; checking if already launched or retrying..."
  sleep 4
done
if [ "$LAUNCH_OK" -ne 1 ]; then
  echo "[Error] Failed to launch detached job. Stopping VM."
  exit 1
fi

echo "=== [4/5] Polling detached job (hard limit=${MAX_WALL_SEC}s) ==="
START_TS=$(date +%s)
CONSEC_FAIL=0
STATUS="TIMEOUT"

cat << 'PYEOF' > /tmp/poll_wave0b.py
import pathlib
w = pathlib.Path("/content/work/wave0b")
if (w / "DONE.flag").exists():
    print("STATE:DONE")
elif (w / "FAIL.flag").exists():
    print("STATE:FAIL:" + (w / "FAIL.flag").read_text().strip())
else:
    print("STATE:RUNNING")
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
  if POLL_OUT=$(timeout 30 "$COLAB" --auth=oauth2 exec -s "$SESSION" --timeout 20 -f /tmp/poll_wave0b.py 2>&1); then
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
timeout 60 "$COLAB" --auth=oauth2 download -s "$SESSION" /content/work/wave0b/job.log "$OUT_LOCAL_DIR/wave0b_job.log" || true
if [ "$STATUS" = "DONE" ]; then
  timeout 120 "$COLAB" --auth=oauth2 download -s "$SESSION" /content/work/wave0b_outputs.tar.gz "$OUT_LOCAL_DIR/wave0b_outputs.tar.gz"
  tar -xzf "$OUT_LOCAL_DIR/wave0b_outputs.tar.gz" -C "$OUT_LOCAL_DIR"
  rm -f "$OUT_LOCAL_DIR/wave0b_outputs.tar.gz"
  ls -lh "$OUT_LOCAL_DIR"
else
  echo "[Error] Job ended with STATUS=$STATUS. Check $OUT_LOCAL_DIR/wave0b_job.log"
  tail -n 30 "$OUT_LOCAL_DIR/wave0b_job.log" 2>/dev/null || true
  exit 1
fi
