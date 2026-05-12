"""
expert_post_check.hardware_check
================================
Probes available hardware and decides whether to run the TRM-slot reasoner
and the 6-phase pipeline in PARALLEL (multi-core / GPU present) or SERIAL
(single-core CPU-only / low RAM).

Decision matrix
---------------
  PARALLEL  — GPU detected via torch.cuda  OR  logical CPUs >= 4 AND RAM >= 8 GB
  SERIAL    — anything else
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class PostCheckMode(str, Enum):
    PARALLEL = "parallel"
    SERIAL = "serial"


@dataclass
class HardwareProfile:
    has_gpu: bool
    gpu_name: Optional[str]
    cpu_count: int
    ram_gb: float
    mode: PostCheckMode
    notes: list = field(default_factory=list)


def probe_hardware() -> HardwareProfile:
    """Return a HardwareProfile describing the current machine and the
    recommended PostCheckMode.

    Never raises — falls back gracefully when optional libraries are absent.
    """
    has_gpu = False
    gpu_name: Optional[str] = None
    cpu_count = 1
    ram_gb = 0.0
    notes: list = []

    # --- GPU probe (torch) ---
    try:
        import torch
        if torch.cuda.is_available():
            has_gpu = True
            gpu_name = torch.cuda.get_device_name(0)
            notes.append(f"CUDA GPU: {gpu_name}")
        else:
            notes.append("torch present but no CUDA GPU detected")
    except ImportError:
        notes.append("torch not available — skipping GPU probe")

    # --- CPU count ---
    try:
        import os
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    notes.append(f"Logical CPUs: {cpu_count}")

    # --- RAM ---
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        notes.append(f"Total RAM: {ram_gb:.1f} GB")
    except ImportError:
        notes.append("psutil not available — RAM unknown, assuming 4 GB")
        ram_gb = 4.0

    # --- Decision ---
    if has_gpu or (cpu_count >= 4 and ram_gb >= 8.0):
        mode = PostCheckMode.PARALLEL
    else:
        mode = PostCheckMode.SERIAL

    profile = HardwareProfile(
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        cpu_count=cpu_count,
        ram_gb=round(ram_gb, 2),
        mode=mode,
        notes=notes,
    )
    logger.info(
        "Hardware probe complete — mode=%s | GPU=%s | CPUs=%d | RAM=%.1f GB",
        mode.value, gpu_name or "none", cpu_count, ram_gb,
    )
    return profile
