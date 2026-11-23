"""Colab GPU routines for ESM-2 masked-marginals, fp16 diff embeddings, and LoRA / LoRA-LLR.

Note: `torch`, `transformers`, and `peft` are imported lazily inside functions so this module
remains importable in the local CPU-only environment.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np
import pandas as pd

from esm2_ood.metrics import compute_mse, compute_spearman
from esm2_ood.splits import esm2_token_index

MODEL_IDS = {
    "8M": "facebook/esm2_t6_8M_UR50D",
    "35M": "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    "3B": "facebook/esm2_t36_3B_UR50D",
}


def compute_esm2_masked_marginals(
    df: pd.DataFrame,
    wt_seq: str,
    model_name_or_size: str = "650M",
    batch_size: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    """Compute official ProteinGym ESM-2 masked-marginals LLR scores.

    Masks each unique mutated position once and scores all substitutions at that position.
    Uses token index `pos` (1-based) directly, skipping `<cls>` at token index 0.
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    model_id = MODEL_IDS.get(model_name_or_size, model_name_or_size)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    enc = tokenizer(wt_seq, return_tensors="pt", add_special_tokens=True)
    base_ids = enc["input_ids"][0]
    attn_mask = enc["attention_mask"][0]
    mask_token_id = tokenizer.mask_token_id

    unique_positions = sorted({int(p) for p in df["pos"]})
    pos_to_logprobs: dict[int, np.ndarray] = {}

    with torch.no_grad():
        for start in range(0, len(unique_positions), batch_size):
            batch_pos = unique_positions[start : start + batch_size]
            batch_ids = base_ids.unsqueeze(0).repeat(len(batch_pos), 1).to(device)
            batch_mask = attn_mask.unsqueeze(0).repeat(len(batch_pos), 1).to(device)
            for b_idx, pos in enumerate(batch_pos):
                tok_idx = esm2_token_index(pos, wt_seq=wt_seq)
                batch_ids[b_idx, tok_idx] = mask_token_id

            logits = model(input_ids=batch_ids, attention_mask=batch_mask).logits
            for b_idx, pos in enumerate(batch_pos):
                tok_idx = esm2_token_index(pos, wt_seq=wt_seq)
                log_probs = torch.log_softmax(logits[b_idx, tok_idx].float(), dim=-1)
                pos_to_logprobs[pos] = log_probs.cpu().numpy()

    scores = np.zeros(len(df), dtype=np.float64)
    for row_idx, (wt_aa, pos, mut_aa) in enumerate(zip(df["wt_aa"], df["pos"], df["mut_aa"])):
        tok_idx = esm2_token_index(int(pos), wt_seq=wt_seq, expected_wt=wt_aa)
        _ = tok_idx
        wt_id = tokenizer.convert_tokens_to_ids(wt_aa)
        mut_id = tokenizer.convert_tokens_to_ids(mut_aa)
        lp = pos_to_logprobs[int(pos)]
        scores[row_idx] = float(lp[mut_id] - lp[wt_id])

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def extract_frozen_diff_embeddings(
    df: pd.DataFrame,
    wt_seq: str,
    model_name_or_size: str = "650M",
    batch_size: int = 16,
    device: str = "cuda",
) -> np.ndarray:
    """Extract fp16 last-hidden-state difference vectors `h_mut[pos] - h_wt[pos]`."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    model_id = MODEL_IDS.get(model_name_or_size, model_name_or_size)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype).to(device)
    model.eval()

    wt_enc = tokenizer(wt_seq, return_tensors="pt", add_special_tokens=True)
    wt_ids = wt_enc["input_ids"].to(device)
    wt_mask = wt_enc["attention_mask"].to(device)

    with torch.no_grad():
        wt_hidden = model(input_ids=wt_ids, attention_mask=wt_mask).last_hidden_state[0].float().cpu()

    hidden_dim = wt_hidden.shape[-1]
    diffs = np.zeros((len(df), hidden_dim), dtype=np.float16)

    mut_seqs = df["mutated_sequence"].tolist()
    positions = df["pos"].tolist()
    wt_aas = df["wt_aa"].tolist()

    with torch.no_grad():
        for start in range(0, len(df), batch_size):
            batch_seqs = mut_seqs[start : start + batch_size]
            batch_pos = positions[start : start + batch_size]
            batch_wt = wt_aas[start : start + batch_size]

            enc = tokenizer(batch_seqs, return_tensors="pt", padding=False, add_special_tokens=True)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float().cpu()

            for i, (pos, wt_aa) in enumerate(zip(batch_pos, batch_wt)):
                tok_idx = esm2_token_index(int(pos), wt_seq=wt_seq, expected_wt=wt_aa)
                diff_vec = out[i, tok_idx] - wt_hidden[tok_idx]
                diffs[start + i] = diff_vec.numpy().astype(np.float16)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return diffs


def _fit_ols_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit closed-form 1D OLS `y ≈ slope * x + intercept` with zero-variance guard."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) < 2:
        return 1.0, 0.0
    x_mean = float(np.mean(x_arr))
    y_mean = float(np.mean(y_arr))
    dx = x_arr - x_mean
    var_x = float(np.mean(dx * dx))
    if var_x <= 1e-12:
        return 1.0, 0.0
    cov_xy = float(np.mean(dx * (y_arr - y_mean)))
    slope = cov_xy / var_x
    intercept = y_mean - slope * x_mean
    return float(slope), float(intercept)


def train_eval_esm2_lora_llr(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wt_seq: str,
    model_name_or_size: str = "650M",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lr: float = 1e-4,
    max_epochs: int = 12,
    patience: int = 3,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fine-tune ESM-2 with LoRA on (query, value) using LoRA-LLR readout (masked lm_head).

    Initializes (scale_param, bias_param) via train-only closed-form OLS on epoch-0 LLR
    before LoRA optimization so MSE starts calibrated. Every fold initializes a fresh
    PeftModel from scratch; no adapter state leaks across folds.
    """
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model_id = MODEL_IDS.get(model_name_or_size, model_name_or_size)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForMaskedLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    )

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_model, lora_cfg).to(device)

    # Train-only target standardization
    y_tr_raw = train_df["DMS_score"].to_numpy(dtype=np.float64)
    y_mean = float(np.mean(y_tr_raw))
    y_std = float(np.std(y_tr_raw)) or 1.0
    y_tr_norm = (y_tr_raw - y_mean) / y_std

    # Affine calibration parameters on top of LLR for MSE alignment
    scale_param = torch.nn.Parameter(torch.tensor(1.0, device=device))
    bias_param = torch.nn.Parameter(torch.tensor(0.0, device=device))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + [scale_param, bias_param],
        lr=lr,
        weight_decay=1e-2,
    )

    wt_enc = tokenizer(wt_seq, return_tensors="pt", add_special_tokens=True)
    wt_ids_base = wt_enc["input_ids"][0]
    wt_mask_base = wt_enc["attention_mask"][0]
    mask_token_id = tokenizer.mask_token_id

    def _compute_raw_llr(sub_df: pd.DataFrame) -> np.ndarray:
        model.eval()
        llrs = np.zeros(len(sub_df), dtype=np.float64)
        uniq_pos = sorted({int(p) for p in sub_df["pos"]})
        pos_lp = {}
        with torch.no_grad():
            for s in range(0, len(uniq_pos), batch_size * 2):
                bp = uniq_pos[s : s + batch_size * 2]
                b_ids = wt_ids_base.unsqueeze(0).repeat(len(bp), 1).to(device)
                b_mask = wt_mask_base.unsqueeze(0).repeat(len(bp), 1).to(device)
                for idx_b, p in enumerate(bp):
                    b_ids[idx_b, esm2_token_index(p, wt_seq=wt_seq)] = mask_token_id
                logits = model(input_ids=b_ids, attention_mask=b_mask).logits
                for idx_b, p in enumerate(bp):
                    t_idx = esm2_token_index(p, wt_seq=wt_seq)
                    pos_lp[p] = torch.log_softmax(logits[idx_b, t_idx].float(), dim=-1).cpu()
        for r_i, (wt_a, p, mut_a) in enumerate(zip(sub_df["wt_aa"], sub_df["pos"], sub_df["mut_aa"])):
            lp = pos_lp[int(p)]
            llrs[r_i] = float(lp[tokenizer.convert_tokens_to_ids(mut_a)] - lp[tokenizer.convert_tokens_to_ids(wt_a)])
        return llrs

    def _predict_df(sub_df: pd.DataFrame) -> np.ndarray:
        llrs = _compute_raw_llr(sub_df)
        s_val = float(scale_param.detach().cpu())
        b_val = float(bias_param.detach().cpu())
        return (s_val * llrs + b_val) * y_std + y_mean

    # Epoch-0 train-only closed-form OLS calibration before LoRA optimization
    llr_tr0 = _compute_raw_llr(train_df)
    s0, b0 = _fit_ols_affine(llr_tr0, y_tr_norm)
    with torch.no_grad():
        scale_param.fill_(s0)
        bias_param.fill_(b0)

    best_val_rho = -float("inf")
    best_state: dict[str, Any] | None = None
    no_improve = 0

    tr_rows = list(zip(train_df["wt_aa"], train_df["pos"], train_df["mut_aa"], y_tr_norm))
    rng = np.random.default_rng(seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(len(tr_rows))
        for s in range(0, len(perm), batch_size):
            batch_idx = perm[s : s + batch_size]
            b_ids = wt_ids_base.unsqueeze(0).repeat(len(batch_idx), 1).to(device)
            b_mask = wt_mask_base.unsqueeze(0).repeat(len(batch_idx), 1).to(device)
            t_indices = []
            wt_toks = []
            mut_toks = []
            targets = []
            for b_i, row_i in enumerate(batch_idx):
                wt_a, p, mut_a, y_norm = tr_rows[int(row_i)]
                t_idx = esm2_token_index(int(p), wt_seq=wt_seq, expected_wt=wt_a)
                b_ids[b_i, t_idx] = mask_token_id
                t_indices.append(t_idx)
                wt_toks.append(tokenizer.convert_tokens_to_ids(wt_a))
                mut_toks.append(tokenizer.convert_tokens_to_ids(mut_a))
                targets.append(y_norm)

            logits = model(input_ids=b_ids, attention_mask=b_mask).logits
            llrs = []
            for b_i, (t_idx, wt_id, mut_id) in enumerate(zip(t_indices, wt_toks, mut_toks)):
                lp = torch.log_softmax(logits[b_i, t_idx].float(), dim=-1)
                llrs.append(lp[mut_id] - lp[wt_id])
            llr_tensor = torch.stack(llrs)
            y_pred_norm = scale_param * llr_tensor + bias_param
            y_tgt = torch.tensor(targets, dtype=torch.float32, device=device)
            loss = F.mse_loss(y_pred_norm, y_tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_pred = _predict_df(val_df)
        val_rho = compute_spearman(val_df["DMS_score"].to_numpy(), val_pred)
        val_score = val_rho if val_rho is not None else -float("inf")
        if val_score > best_val_rho:
            best_val_rho = val_score
            best_state = {
                "lora": {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k},
                "scale": float(scale_param.detach().cpu()),
                "bias": float(bias_param.detach().cpu()),
                "best_epoch": epoch,
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["lora"], strict=False)
        with torch.no_grad():
            scale_param.fill_(best_state["scale"])
            bias_param.fill_(best_state["bias"])

    test_pred = _predict_df(test_df)
    elapsed_sec = time.perf_counter() - t0
    test_rho = compute_spearman(test_df["DMS_score"].to_numpy(), test_pred)
    test_mse = compute_mse(test_df["DMS_score"].to_numpy(), test_pred)

    del model, base_model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "test_pred": test_pred,
        "test_spearman": test_rho,
        "test_mse": test_mse,
        "best_val_spearman": best_val_rho,
        "best_epoch": best_state["best_epoch"] if best_state else 0,
        "elapsed_sec": elapsed_sec,
    }


def train_eval_esm2_lora_linear_head(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    wt_seq: str,
    model_name_or_size: str = "650M",
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lr: float = 1e-4,
    max_epochs: int = 12,
    patience: int = 3,
    batch_size: int = 8,
    seed: int = 42,
    device: str = "cuda",
) -> dict[str, Any]:
    """Fine-tune ESM-2 with LoRA on (query, value) using a linear head on site-diff features.

    Isomorphic to FrozenDiffRidgeBaseline: uses `h_mut[pos] - h_wt[pos]` from the last
    hidden state followed by a linear projection head. Every fold initializes a fresh
    PeftModel and linear head from scratch.
    """
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer

    t0 = time.perf_counter()
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    model_id = MODEL_IDS.get(model_name_or_size, model_name_or_size)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModel.from_pretrained(
        model_id, torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32
    )

    lora_cfg = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["query", "value"],
    )
    model = get_peft_model(base_model, lora_cfg).to(device)
    hidden_dim = int(base_model.config.hidden_size)
    head = torch.nn.Linear(hidden_dim, 1).to(device)

    y_tr_raw = train_df["DMS_score"].to_numpy(dtype=np.float64)
    y_mean = float(np.mean(y_tr_raw))
    y_std = float(np.std(y_tr_raw)) or 1.0
    y_tr_norm = (y_tr_raw - y_mean) / y_std

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
        lr=lr,
        weight_decay=1e-2,
    )

    wt_enc = tokenizer(wt_seq, return_tensors="pt", add_special_tokens=True)
    wt_ids = wt_enc["input_ids"].to(device)
    wt_mask = wt_enc["attention_mask"].to(device)

    def _predict_df(sub_df: pd.DataFrame) -> np.ndarray:
        model.eval()
        head.eval()
        preds = np.zeros(len(sub_df), dtype=np.float64)
        mut_seqs = sub_df["mutated_sequence"].tolist()
        positions = sub_df["pos"].tolist()
        wt_aas = sub_df["wt_aa"].tolist()
        with torch.no_grad():
            wt_hidden = model(input_ids=wt_ids, attention_mask=wt_mask).last_hidden_state[0].float()
            for s in range(0, len(sub_df), batch_size * 2):
                b_seqs = mut_seqs[s : s + batch_size * 2]
                b_pos = positions[s : s + batch_size * 2]
                b_wt = wt_aas[s : s + batch_size * 2]
                enc = tokenizer(b_seqs, return_tensors="pt", padding=False, add_special_tokens=True)
                b_ids = enc["input_ids"].to(device)
                b_mask = enc["attention_mask"].to(device)
                out = model(input_ids=b_ids, attention_mask=b_mask).last_hidden_state.float()
                diffs = []
                for i, (pos, wt_aa) in enumerate(zip(b_pos, b_wt)):
                    t_idx = esm2_token_index(int(pos), wt_seq=wt_seq, expected_wt=wt_aa)
                    diffs.append(out[i, t_idx] - wt_hidden[t_idx])
                diff_tensor = torch.stack(diffs)
                y_norm = head(diff_tensor).squeeze(-1).cpu().numpy()
                preds[s : s + len(b_seqs)] = y_norm * y_std + y_mean
        return preds

    best_val_rho = -float("inf")
    best_state: dict[str, Any] | None = None
    no_improve = 0

    tr_rows = list(zip(train_df["mutated_sequence"], train_df["wt_aa"], train_df["pos"], y_tr_norm))
    rng = np.random.default_rng(seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        head.train()
        perm = rng.permutation(len(tr_rows))
        for s in range(0, len(perm), batch_size):
            batch_idx = perm[s : s + batch_size]
            b_seqs = [tr_rows[int(i)][0] for i in batch_idx]
            b_wt = [tr_rows[int(i)][1] for i in batch_idx]
            b_pos = [tr_rows[int(i)][2] for i in batch_idx]
            targets = [tr_rows[int(i)][3] for i in batch_idx]

            wt_hidden = model(input_ids=wt_ids, attention_mask=wt_mask).last_hidden_state[0].float()
            enc = tokenizer(b_seqs, return_tensors="pt", padding=False, add_special_tokens=True)
            b_ids = enc["input_ids"].to(device)
            b_mask = enc["attention_mask"].to(device)
            out = model(input_ids=b_ids, attention_mask=b_mask).last_hidden_state.float()

            diffs = []
            for i, (pos, wt_aa) in enumerate(zip(b_pos, b_wt)):
                t_idx = esm2_token_index(int(pos), wt_seq=wt_seq, expected_wt=wt_aa)
                diffs.append(out[i, t_idx] - wt_hidden[t_idx])
            diff_tensor = torch.stack(diffs)
            y_pred_norm = head(diff_tensor).squeeze(-1)
            y_tgt = torch.tensor(targets, dtype=torch.float32, device=device)
            loss = F.mse_loss(y_pred_norm, y_tgt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad] + list(head.parameters()), 1.0
            )
            optimizer.step()

        val_pred = _predict_df(val_df)
        val_rho = compute_spearman(val_df["DMS_score"].to_numpy(), val_pred)
        val_score = val_rho if val_rho is not None else -float("inf")
        if val_score > best_val_rho:
            best_val_rho = val_score
            best_state = {
                "lora": {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                "best_epoch": epoch,
            }
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state["lora"], strict=False)
        head.load_state_dict(best_state["head"])

    test_pred = _predict_df(test_df)
    elapsed_sec = time.perf_counter() - t0
    test_rho = compute_spearman(test_df["DMS_score"].to_numpy(), test_pred)
    test_mse = compute_mse(test_df["DMS_score"].to_numpy(), test_pred)

    del model, base_model, head
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return {
        "test_pred": test_pred,
        "test_spearman": test_rho,
        "test_mse": test_mse,
        "best_val_spearman": best_val_rho,
        "best_epoch": best_state["best_epoch"] if best_state else 0,
        "elapsed_sec": elapsed_sec,
    }
