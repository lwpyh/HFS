"""
HFS (Hierarchical Frame Selection) — inference-time video frame selection.

Algorithm: UCB segment budget allocation + MMR per-segment frame selection,
guided by CLIP text-image similarity derived from the question.

Pipeline
--------
1. Sample 512 candidate frames uniformly from the video
2. Score each frame with CLIP cosine similarity to the question text
3. Divide into 24 temporal segments; allocate 128 budget tokens via UCB
4. Within each segment, select diverse, relevant frames via MMR
5. Feed the 128 selected frames to the VLM for inference
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
MAX_PIXELS = 1605632  # matches official lmms-eval Qwen2.5-VL default
MIN_PIXELS = 256 * 28 * 28
_SYS = "You are a helpful assistant."

# ── METHOD_REGISTRY: populated at bottom ──────────────────────────────────────
METHOD_REGISTRY: Dict[str, object] = {}


# ── Video I/O ──────────────────────────────────────────────────────────────────


def _uniform_frames_times(vpath: str, n: int = 32):
    """Returns (frames, duration, timestamps).  Uses cv2 for broad format support."""
    import cv2

    cap = cv2.VideoCapture(vpath)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 1.0
    duration = total / fps
    if total <= 0:
        cap.release()
        return [], duration, []
    indices = [
        min(int(round(i * (total - 1) / max(n - 1, 1))), total - 1) for i in range(n)
    ]
    frames, times = [], []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        times.append(idx / fps)
    cap.release()
    return frames, duration, times


# ── Output helpers ─────────────────────────────────────────────────────────────


def _frames_only_out(frames: List[Image.Image]) -> dict:
    """Return PIL frames — official benchmark prompt is preserved downstream."""
    content = [
        {"type": "image", "image": img, "max_pixels": MAX_PIXELS, "min_pixels": MIN_PIXELS}
        for img in frames
    ]
    return {"messages": [{"role": "system", "content": _SYS}, {"role": "user", "content": content}]}


# ── Query compaction ───────────────────────────────────────────────────────────


def _compact_query_text(question: str, max_chars: int = 320) -> str:
    text = re.sub(r"\s+", " ", question).strip()
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    return clipped.rsplit(" ", 1)[0] if " " in clipped else clipped


# ── CLIP scoring ───────────────────────────────────────────────────────────────


def _minmax_norm(values) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return np.full_like(values, 0.5, dtype=np.float32)
    return (values - lo) / (hi - lo + 1e-6)


@lru_cache(maxsize=4)
def _load_clip_cached(model_name: str):
    import torch
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.to(dtype=torch.float16)
    return model, processor, device


def _get_clip(preferred: Optional[str] = None):
    candidates = [preferred, "openai/clip-vit-large-patch14", "openai/clip-vit-base-patch32"]
    last_err = None
    for name in candidates:
        if not name:
            continue
        try:
            return _load_clip_cached(name) + (name,)
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"No CLIP checkpoint found: {last_err}")


def _clip_score_frames(
    frames: List[Image.Image],
    query_text: str,
    model_name: Optional[str] = None,
    batch_size: int = 16,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (image_features [N, D], relevance_scores [N])."""
    import torch
    import torch.nn.functional as F

    clip_model, clip_processor, device, _ = _get_clip(model_name)
    image_features = []
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            batch = list(frames[start : start + batch_size])
            inputs = clip_processor(images=batch, return_tensors="pt")
            pv = inputs["pixel_values"].to(device)
            if device == "cuda":
                pv = pv.to(dtype=torch.float16)
            raw_img = clip_model.get_image_features(pixel_values=pv)
            if not isinstance(raw_img, torch.Tensor):
                raw_img = (
                    raw_img.pooler_output if hasattr(raw_img, "pooler_output") else raw_img[1]
                )
            feats = F.normalize(raw_img.float(), dim=-1)
            image_features.append(feats.cpu())
        text_in = clip_processor(
            text=[query_text], return_tensors="pt", padding=True, truncation=True
        )
        text_in = {k: v.to(device) for k, v in text_in.items()}
        raw_txt = clip_model.get_text_features(**text_in)
        if not isinstance(raw_txt, torch.Tensor):
            raw_txt = (
                raw_txt.pooler_output if hasattr(raw_txt, "pooler_output") else raw_txt[1]
            )
        text_feat = F.normalize(raw_txt.float(), dim=-1)[0].cpu()
    img_feats = torch.cat(image_features, dim=0).numpy().astype(np.float32)
    text_feat = text_feat.numpy().astype(np.float32)
    return img_feats, (img_feats @ text_feat).astype(np.float32)


# ── UCB + MMR core ─────────────────────────────────────────────────────────────


def _contiguous_groups(n_items: int, n_groups: int) -> List[List[int]]:
    n_groups = max(1, min(n_groups, n_items))
    edges = np.linspace(0, n_items, n_groups + 1, dtype=int)
    groups = []
    for s, e in zip(edges[:-1], edges[1:]):
        if e > s:
            groups.append(list(range(int(s), int(e))))
    return groups


def _ucb_quotas(
    relevance: np.ndarray,
    total_k: int = 32,
    n_segs: int = 12,
    beta: float = 0.35,
):
    """Allocate total_k budget tokens across temporal segments via UCB.

    s_j = (μ_j + β × σ_j) / √(k_j + 1)

    where μ_j = mean of top-2 relevance scores in segment j,
    σ_j = std of relevance scores in segment j (exploration bonus),
    k_j = current allocation (diminishing returns penalty).
    """
    groups = _contiguous_groups(len(relevance), n_segs)
    if not groups:
        return [], np.zeros(0, dtype=np.int32)
    total_k = min(total_k, len(relevance))
    quotas = np.zeros(len(groups), dtype=np.int32)
    base = min(total_k, len(groups))
    quotas[:base] = 1
    remaining = total_k - int(quotas.sum())
    rel = _minmax_norm(relevance)
    seg_stats = []
    for grp in groups:
        vals = rel[grp]
        top_m = np.sort(vals)[-min(2, len(vals)) :]
        seg_stats.append((
            float(top_m.mean()),
            float(vals.std()) if len(vals) > 1 else 0.0,
        ))
    for _ in range(max(0, remaining)):
        best_j, best_score = 0, -1e9
        for j, (mean_top, std) in enumerate(seg_stats):
            score = (mean_top + beta * std) / math.sqrt(float(quotas[j]) + 1.0)
            if score > best_score:
                best_score, best_j = score, j
        quotas[best_j] += 1
    return groups, quotas


def _select_mmr(
    relevance: np.ndarray,
    img_feats: np.ndarray,
    k: int = 32,
    alpha: float = 0.72,
    cands: Optional[Sequence[int]] = None,
) -> List[int]:
    """Greedy MMR frame selection.

    MMR(f_i) = α × r_i − (1 − α) × max_{f_i' ∈ selected} cosine_sim(f_i, f_i')
    """
    cands = list(cands) if cands is not None else list(range(len(relevance)))
    if not cands:
        return []
    if k >= len(cands):
        return sorted(int(i) for i in cands)
    local_rel = _minmax_norm(np.asarray(relevance)[cands])
    local_feats = np.asarray(img_feats)[cands].astype(np.float32)
    pairwise = local_feats @ local_feats.T
    first = int(np.argmax(local_rel))
    sel = [first]
    rem = set(range(len(cands))) - {first}
    while rem and len(sel) < k:
        rem_arr = np.array(sorted(rem), dtype=np.int32)
        red = pairwise[rem_arr][:, sel].max(axis=1)
        mmr = alpha * local_rel[rem_arr] - (1.0 - alpha) * red
        pick = int(rem_arr[int(np.argmax(mmr))])
        sel.append(pick)
        rem.remove(pick)
    return sorted(int(cands[i]) for i in sel)


def _hfs_select(
    frames: List[Image.Image],
    question: str,
    out_k: int = 128,
    n_segs: int = 24,
    beta: float = 0.35,
    alpha: float = 0.72,
) -> List[int]:
    """Run the UCB+MMR selection pipeline. Returns sorted frame indices."""
    query = _compact_query_text(question)
    img_feats, relevance = _clip_score_frames(frames, query)
    target_k = min(out_k, len(frames))
    groups, quotas = _ucb_quotas(
        relevance,
        total_k=target_k,
        n_segs=min(n_segs, len(frames)),
        beta=beta,
    )
    selected, used = [], set()
    for grp, quota in zip(groups, quotas):
        if quota <= 0:
            continue
        local = _select_mmr(
            relevance, img_feats,
            k=min(int(quota), len(grp)),
            alpha=alpha,
            cands=grp,
        )
        for i in local:
            if i not in used:
                used.add(i)
                selected.append(i)
    if len(selected) < target_k:
        leftover = [i for i in range(len(frames)) if i not in used]
        refill = _select_mmr(
            relevance, img_feats,
            k=target_k - len(selected),
            alpha=alpha,
            cands=leftover,
        )
        selected.extend(refill)
    return sorted(selected[:target_k])


# ── Method classes ─────────────────────────────────────────────────────────────


class Uniform128:
    """Baseline: uniform 128 frames, official benchmark prompt."""

    name = "uniform_128"
    tags = ["videomme"]

    def run(self, vpath: str, question: str, **kwargs) -> dict:
        try:
            frames, _, _ = _uniform_frames_times(vpath, 128)
            if not frames:
                return {"messages": []}
            return _frames_only_out(frames)
        except Exception as e:
            from loguru import logger as _lg
            _lg.warning(f"uniform_128 failed: {e}")
            return {"messages": []}


class HFS:
    """HFS: UCB+MMR selects 128 frames from 512 candidates, official benchmark prompt.

    Hyper-parameters
    ----------------
    _CAND_N = 512   candidate frames sampled uniformly
    _OUT_K  = 128   frames fed to the VLM
    _N_SEGS = 24    temporal segments for UCB allocation
    _BETA   = 0.35  UCB exploration bonus (variation weight)
    _ALPHA  = 0.72  MMR relevance-diversity trade-off
    """

    name = "hfs"
    tags = ["clip", "ucb", "mmr", "videomme"]

    _CAND_N = 512
    _OUT_K = 128
    _N_SEGS = 24
    _BETA = 0.35
    _ALPHA = 0.72

    def run(self, vpath: str, question: str, **kwargs) -> dict:
        try:
            frames, _, _ = _uniform_frames_times(vpath, self._CAND_N)
            if not frames:
                return {"messages": []}
            selected = _hfs_select(
                frames, question,
                out_k=self._OUT_K, n_segs=self._N_SEGS,
                beta=self._BETA, alpha=self._ALPHA,
            )
            return _frames_only_out([frames[i] for i in selected])
        except Exception as e:
            from loguru import logger as _lg
            _lg.warning(f"HFS failed: {e}")
            return {"messages": []}


# ── Register methods ───────────────────────────────────────────────────────────

METHOD_REGISTRY["uniform_128"] = Uniform128()
METHOD_REGISTRY["hfs"] = HFS()

# Legacy aliases
METHOD_REGISTRY["uniform_128_direct"] = METHOD_REGISTRY["uniform_128"]
METHOD_REGISTRY["hierarchical_focus_128_direct"] = METHOD_REGISTRY["hfs"]
