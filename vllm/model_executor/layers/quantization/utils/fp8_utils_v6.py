"""
V6 Kernel - Hybrid Dispatch with Aggressive Decode Optimization
Adds smaller tiling for tiny M, explicit prefill fallback to baseline.
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
# V6 Configuration Constants - More aggressive tiling for tiny M
# ============================================================================
# Tuple: (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)

# Decode shapes (M, 2048, 7168)
_V6_CONFIG_DECODE_MICRO = (8, 64, 128, 4, 4, 4)     # M <= 2 - More aggressive
_V6_CONFIG_DECODE_TINY = (16, 64, 128, 4, 4, 4)     # M <= 4
_V6_CONFIG_DECODE_SMALL = (32, 64, 128, 4, 4, 4)    # M <= 8
_V6_CONFIG_DECODE_MEDIUM = (32, 128, 128, 4, 4, 4)  # M <= 16
_V6_CONFIG_DECODE_DEFAULT = (64, 128, 128, 8, 4, 4) # M > 16

# Square shapes (M, 7168, 7168)
_V6_CONFIG_SQUARE_MICRO = (8, 32, 128, 4, 4, 4)     # M <= 2 - More aggressive
_V6_CONFIG_SQUARE_TINY = (16, 32, 128, 4, 4, 4)     # M <= 4
_V6_CONFIG_SQUARE_SMALL = (32, 64, 128, 4, 4, 4)    # M <= 8
_V6_CONFIG_SQUARE_MEDIUM = (32, 128, 128, 4, 4, 4)  # M <= 16
_V6_CONFIG_SQUARE_DEFAULT = (64, 128, 128, 8, 4, 4) # M > 16


@functools.lru_cache(maxsize=256)
def get_v6_config(M: int, N: int, K: int) -> Optional[Tuple]:
    """
    V6 optimized configs with hybrid dispatch.
    
    Key difference from V5:
    - More aggressive tiling for tiny M (uses BLOCK_M=8 for M<=2)
    - Explicit prefill fallback (N > K * 1.5 always uses baseline)
    
    NOTE: BLOCK_M must be >= M for correctness. For M < 4, fallback to baseline.
    """
    if not _use_optimized_mi355x_kernel():
        return None
    
    # HYBRID DISPATCH: For prefill shapes (N > K * 1.5), use baseline
    if N > K * 1.5:
        return None  # Fall back to baseline for prefill-like shapes
    
    # For very tiny M (< 4), BLOCK_M=8 causes issues, use baseline
    if M < 4:
        return None
    
    # Decode shapes (M, 2048, 7168)
    if N <= 2100 and K >= 7000:
        if M <= 4:
            return _V6_CONFIG_DECODE_TINY
        elif M <= 8:
            return _V6_CONFIG_DECODE_SMALL
        elif M <= 16:
            return _V6_CONFIG_DECODE_MEDIUM
        else:
            return _V6_CONFIG_DECODE_DEFAULT
    
    # Square shapes (M, 7168, 7168)
    if N >= 7000 and K >= 7000 and abs(N - K) <= 200:
        if M <= 4:
            return _V6_CONFIG_SQUARE_TINY
        elif M <= 8:
            return _V6_CONFIG_SQUARE_SMALL
        elif M <= 16:
            return _V6_CONFIG_SQUARE_MEDIUM
        else:
            return _V6_CONFIG_SQUARE_DEFAULT
    
    return None
