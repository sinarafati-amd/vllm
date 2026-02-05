"""
V5 Kernel - Original MI355X optimizations for decode/square shapes
First optimized version with basic decode/square shape tuning.
"""

import functools
from typing import Optional, Tuple

from vllm.platforms import current_platform


@functools.lru_cache(maxsize=1)
def _use_optimized_mi355x_kernel() -> bool:
    """Check if we should use the optimized MI355X kernel. Result is cached."""
    if not current_platform.is_rocm():
        return False
    device_name = current_platform.get_device_name()
    return "MI355" in device_name or "MI300" in device_name or "MI325" in device_name


# ============================================================================
# V5 Configuration Constants
# ============================================================================
# Tuple: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)

# Decode shapes (M, 2048, 7168) - K > N pattern
_V5_CONFIG_DECODE_MICRO = (16, 128, 128, 4, 4, 4)   # M <= 2
_V5_CONFIG_DECODE_TINY = (16, 64, 128, 4, 4, 4)     # M <= 4
_V5_CONFIG_DECODE_SMALL = (32, 64, 128, 4, 4, 4)    # M <= 8
_V5_CONFIG_DECODE_MEDIUM = (32, 128, 128, 4, 4, 4)  # M <= 16
_V5_CONFIG_DECODE_DEFAULT = (64, 128, 128, 8, 4, 4) # M > 16

# Square shapes (M, 7168, 7168)
_V5_CONFIG_SQUARE_MICRO = (16, 128, 128, 4, 4, 4)   # M <= 2
_V5_CONFIG_SQUARE_TINY = (16, 64, 128, 4, 4, 4)     # M <= 4
_V5_CONFIG_SQUARE_SMALL = (32, 64, 128, 4, 4, 4)    # M <= 8
_V5_CONFIG_SQUARE_MEDIUM = (32, 128, 128, 4, 4, 4)  # M <= 16
_V5_CONFIG_SQUARE_DEFAULT = (64, 128, 128, 8, 4, 4) # M > 16


@functools.lru_cache(maxsize=256)
def get_v5_config(M: int, N: int, K: int) -> Optional[Tuple]:
    """
    V5 optimized configs for decode and square shapes only.
    Prefill shapes fall back to baseline (returns None).
    """
    if not _use_optimized_mi355x_kernel():
        return None
    
    # Decode shapes (M, 2048, 7168)
    if N <= 2100 and K >= 7000:
        if M <= 2:
            return _V5_CONFIG_DECODE_MICRO
        elif M <= 4:
            return _V5_CONFIG_DECODE_TINY
        elif M <= 8:
            return _V5_CONFIG_DECODE_SMALL
        elif M <= 16:
            return _V5_CONFIG_DECODE_MEDIUM
        else:
            return _V5_CONFIG_DECODE_DEFAULT
    
    # Square shapes (M, 7168, 7168)
    if N >= 7000 and K >= 7000 and abs(N - K) <= 200:
        if M <= 2:
            return _V5_CONFIG_SQUARE_MICRO
        elif M <= 4:
            return _V5_CONFIG_SQUARE_TINY
        elif M <= 8:
            return _V5_CONFIG_SQUARE_SMALL
        elif M <= 16:
            return _V5_CONFIG_SQUARE_MEDIUM
        else:
            return _V5_CONFIG_SQUARE_DEFAULT
    
    # Prefill shapes: Fall back to None (use baseline)
    return None
