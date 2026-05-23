"""data_framestacking.py — Frame-stacking dataset wrapper for LeWM billiards.

For each time step t in a T-step window, stacks three consecutive frames
[t-2, t-1, t] along the channel dimension:
    single frame : (3, H, W)
    stacked frame: (9, H, W)   ← 3 frames × 3 RGB channels

Episode-boundary padding: if t-2 or t-1 would fall before the episode start,
the earliest available frame in the episode is repeated.

Output dict keys:
    pixels : (T, 9, img_size, img_size)  float32 ImageNet-normalised
    action : (T, action_dim)             float32 (raw, un-normalised)
    state  : (T, state_dim)              float32 (raw, un-normalised)

Usage (matches existing swm.data.load_dataset API):
    ds = FrameStackingDataset(path, num_steps=4, frameskip=1, img_size=96)
    ds.transform = Compose(normalizer_action, normalizer_state)
    sample = ds[0]   # pixels: (4, 9, 96, 96)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import h5py
import hdf5plugin  # noqa: F401 – register HDF5 compression codecs before h5py use
import numpy as np
import torch
import torch.nn.functional as F

# ── ImageNet normalization constants (matches get_img_preprocessor in utils.py) ──
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


class FrameStackingDataset:
    """Wraps a billiards HDF5 file to produce 9-channel stacked-frame samples.

    Parameters
    ----------
    path:
        Path to the HDF5 file (e.g. ``~/.stable-wm/billiards_expert_train.h5``).
    num_steps:
        Number of time steps per sample (``history_size + num_preds``, default 4).
    frameskip:
        Stride between sampled steps (default 1, matching the billiards config).
    img_size:
        Target spatial resolution after resize (default 96).
    transform:
        Optional transform applied to the output dict *after* pixels are stacked.
        Intended for column normalizers (action / state) – must NOT touch pixels.
    keys_to_cache:
        Column names to pre-load into RAM.  Billiards default: ``['action','state']``.
    """

    STACK: int = 3  # number of consecutive frames to stack

    def __init__(
        self,
        path: str | Path,
        num_steps: int = 4,
        frameskip: int = 1,
        img_size: int = 96,
        transform: Optional[Callable] = None,
        keys_to_cache: Optional[list[str]] = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.img_size = img_size
        self.transform = transform

        # span: total frames consumed per sample in the underlying flat index
        self.span = num_steps * frameskip

        self._h5: Optional[h5py.File] = None
        self._cache: dict[str, np.ndarray] = {}

        # Read episode metadata
        with h5py.File(self.path, "r") as f:
            self.lengths: np.ndarray = f["ep_len"][:]
            self.offsets: np.ndarray = f["ep_offset"][:]

            for key in keys_to_cache or []:
                self._cache[key] = f[key][:]

        # Build clip indices identical to stable-worldmodel Dataset base class.
        # Each entry is (episode_idx, intra-episode start position).
        self.clip_indices: list[tuple[int, int]] = [
            (ep, start)
            for ep, length in enumerate(self.lengths)
            if length >= self.span
            for start in range(length - self.span + 1)
        ]

    # ── internal helpers ──────────────────────────────────────────────────────

    def _open(self) -> None:
        """Lazily open the HDF5 file (SWMR mode, large read cache)."""
        if self._h5 is None:
            self._h5 = h5py.File(
                self.path, "r", swmr=True, rdcc_nbytes=256 * 1024 * 1024
            )

    def _load_raw_pixels(self, ep_idx: int, start: int, end: int) -> torch.Tensor:
        """Load raw uint8 pixels for episode positions [start, end).

        Returns
        -------
        Tensor of shape ``(end-start, 3, H, W)`` uint8.
        """
        self._open()
        g_start = int(self.offsets[ep_idx]) + start
        g_end   = int(self.offsets[ep_idx]) + end
        # HDF5 stores pixels as (N, H, W, 3) uint8
        data = self._h5["pixels"][g_start:g_end]
        return torch.from_numpy(np.ascontiguousarray(data)).permute(0, 3, 1, 2)

    @staticmethod
    def _preprocess(frames: torch.Tensor, img_size: int) -> torch.Tensor:
        """Convert raw uint8 frames to normalised float32.

        Parameters
        ----------
        frames:
            Shape ``(N, 3, H, W)`` uint8.
        img_size:
            Target spatial size.

        Returns
        -------
        Shape ``(N, 3, img_size, img_size)`` float32, ImageNet-normalised.
        """
        x = frames.float() / 255.0
        # Bilinear resize: (N, 3, H, W) → (N, 3, img_size, img_size)
        x = F.interpolate(
            x, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
        # ImageNet per-channel normalisation
        mean = _IMAGENET_MEAN[:, None, None]  # (3, 1, 1)
        std  = _IMAGENET_STD[:, None, None]   # (3, 1, 1)
        return (x - mean) / std

    # ── Dataset protocol ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.clip_indices)

    def __getitem__(self, idx: int) -> dict:
        ep_idx, start = self.clip_indices[idx]

        # ── 1. Load the extended raw pixel window ─────────────────────────
        # We need frames at positions [start-2, start-1, start, ..., start+span-1]
        # relative to the episode start.  Clamp at 0 for boundary episodes.
        pre = min(self.STACK - 1, start)   # how many real preceding frames exist (0–2)
        load_start = start - pre            # ≥ 0 always
        load_end   = start + self.span      # exclusive

        raw = self._load_raw_pixels(ep_idx, load_start, load_end)
        # raw shape: (pre + span, 3, H, W) uint8

        # ── 2. Pre-process all loaded frames at once ───────────────────────
        processed = self._preprocess(raw, self.img_size)
        # processed shape: (pre + span, 3, img_size, img_size) float32

        # ── 3. Pad the start if load_start was already 0 ──────────────────
        # pad_needed = how many extra copies of frame-0 to prepend
        pad_needed = (self.STACK - 1) - pre  # 0, 1, or 2
        if pad_needed > 0:
            pad = processed[0:1].expand(pad_needed, -1, -1, -1)
            processed = torch.cat([pad, processed], dim=0)
        # After padding, layout of `processed`:
        #   index 0          → frame at episode position start-2  (or padded)
        #   index 1          → frame at episode position start-1  (or padded)
        #   index STACK-1=2  → frame at episode position start
        #   index STACK-1+i  → frame at episode position start + i  (i=0..span-1)

        # ── 4. Build stacked (9-channel) frames for each time step ─────────
        stacked_frames: list[torch.Tensor] = []
        for i in range(self.num_steps):
            # Centre frame index in `processed` for time step i
            t_c = (self.STACK - 1) + i * self.frameskip
            # Stack [t-2, t-1, t] → (9, img_size, img_size)
            stacked = torch.cat(
                [processed[t_c - 2], processed[t_c - 1], processed[t_c]], dim=0
            )
            stacked_frames.append(stacked)

        pixels_stacked = torch.stack(stacked_frames, dim=0)  # (T, 9, h, w)

        # ── 5. Load action and state ───────────────────────────────────────
        self._open()
        g_start = int(self.offsets[ep_idx]) + start
        g_end   = g_start + self.span

        steps: dict = {"pixels": pixels_stacked}

        for col in ("action", "state"):
            if col in self._cache:
                data = torch.from_numpy(
                    np.ascontiguousarray(self._cache[col][g_start:g_end])
                )
            else:
                data = torch.from_numpy(
                    np.ascontiguousarray(self._h5[col][g_start:g_end])
                )

            if col == "action":
                # Reshape to (num_steps, frameskip * action_dim) – matches HDF5Dataset
                data = data.reshape(self.num_steps, -1)
            else:
                # State: apply frameskip downsampling (matches HDF5Dataset logic)
                data = data[:: self.frameskip]

            steps[col] = data

        # ── 6. Apply column normalizers (action / state) ───────────────────
        if self.transform is not None:
            steps = self.transform(steps)

        return steps

    # ── Compatibility shims (used by get_column_normalizer / setattr calls) ─

    def get_col_data(self, col: str) -> np.ndarray:
        """Return the full flat column array from HDF5 (for normalizer fitting)."""
        if col in self._cache:
            return self._cache[col]
        self._open()
        return self._h5[col][:]

    def get_dim(self, col: str) -> int:
        """Return the feature dimension for ``col`` (product of shape[1:])."""
        data = self.get_col_data(col)
        return int(np.prod(data.shape[1:])) if data.ndim > 1 else 1

    def __repr__(self) -> str:
        return (
            f"FrameStackingDataset("
            f"path={self.path.name!r}, "
            f"clips={len(self)}, "
            f"num_steps={self.num_steps}, "
            f"frameskip={self.frameskip}, "
            f"img_size={self.img_size}, "
            f"stack={self.STACK})"
        )
