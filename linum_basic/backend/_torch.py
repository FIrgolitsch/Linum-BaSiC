"""Torch-only DCT-II / DCT-III implementation and LRU-bounded DCT caches.

Pure-PyTorch module-level infrastructure physically extracted from the
historical single-file ``linum_basic/backend.py``: the three LRU-bounded
DCT caches, the twiddle-factor and orthonormal-DCT-matrix builders, the
FFT-based DCT-II / DCT-III implementations (Lee 1984 / Makhoul 1980), the
DCT-kernel-mode selector read by the compiled ALM step, and the
cache-management helpers.

Per D026/D029, isolating this Torch-only "new capability" code into its own
file is the seam that makes ``backend.py``'s port-vs-capability boundary
reviewable in the future S05 PR chain: every symbol in this module is new
functionality with no equivalent in the upstream 401-line
``pybasic/shading_correction.py``, whereas the NumPy/SciPy DCT surface that
mirrors the upstream port lives in :mod:`linum_basic.backend._numpy`.

This file is a verbatim move of the module-level code from the pre-split
``backend.py`` — no logic changes — so the darkfield regression and vignette
validation evidence (K02) collected against the prior tree carries through
bit-identically.
"""

from __future__ import annotations

import math
import os
from collections import OrderedDict
from typing import Any

__all__ = [
    "_DCT_CACHE_MAXSIZE",
    "_DCT_MATRIX_CACHE",
    "_DCT_TWIDDLE_CACHE",
    "_IDCT_TWIDDLE_CACHE",
    "_get_dct_matrix",
    "_get_dct_twiddles",
    "_get_idct_twiddles",
    "_lru_put",
    "_torch_dct1d",
    "_torch_dctn",
    "_torch_idct1d",
    "_torch_idctn",
    "clear_dct_caches",
    "read_dct_kernel_mode",
]


# ---------------------------------------------------------------------------
# PyTorch DCT-II / DCT-III via FFT  (Lee 1984 / Makhoul 1980)
# ---------------------------------------------------------------------------

# Maximum number of entries retained in each DCT cache below.  Once the bound
# is exceeded the least-recently-used entry is evicted, so long-running batch
# jobs that fit many distinct image sizes cannot leak memory across fits (K10).
# Tune by monkeypatching this constant, or release every cached entry between
# batches via :func:`clear_dct_caches`.
_DCT_CACHE_MAXSIZE: int = 64

# Per-(length, dtype, device) cache of precomputed twiddle factors and scale
# vectors.  Populated lazily on first use; avoids O(n) trig recomputation on
# every DCT call inside the hot ALM loop.  Bounded by LRU eviction
# (see ``_DCT_CACHE_MAXSIZE``).
_DCT_TWIDDLE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
_IDCT_TWIDDLE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()

# Per-(length, device) cache of orthonormal DCT-II matrices for matmul-based
# DCT.  Used by _build_alm_step to replace FFT-based DCT inside torch.compile
# regions (Torchinductor cannot generate Triton code for complex ops).
# Bounded by LRU eviction (see ``_DCT_CACHE_MAXSIZE``).
_DCT_MATRIX_CACHE: OrderedDict[tuple, Any] = OrderedDict()


def clear_dct_caches() -> None:
    """Empty all three backend DCT caches.

    Releases the cached orthonormal DCT-II matrices and the forward/inverse
    twiddle factors held at module scope.  Intended for long-running batch
    jobs that fit many stacks with varying shapes and want to reclaim the
    (already LRU-bounded) memory between independent runs, or whenever a
    guaranteed-cold DCT cache is required.

    The caches are bounded by ``_DCT_CACHE_MAXSIZE`` and evict
    least-recently-used entries automatically, so calling this function is
    optional — it only forces an immediate, full release.
    """
    _DCT_MATRIX_CACHE.clear()
    _DCT_TWIDDLE_CACHE.clear()
    _IDCT_TWIDDLE_CACHE.clear()


def read_dct_kernel_mode() -> str:
    """Return DCT matmul layout mode for the compiled ALM step (default-off lever).

    ``LINUM_BASIC_DCT_KERNEL`` selects the matmul path inside ``_build_alm_step``.
    Only ``"tuned"`` enables the contiguous-layout variant; any other value (including
    unset) preserves the production default.

    Returns
    -------
    str
        ``"tuned"`` when the env var is set to ``tuned``; otherwise ``"default"``.
    """
    raw = os.environ.get("LINUM_BASIC_DCT_KERNEL", "default").strip().lower()
    return "tuned" if raw == "tuned" else "default"


def _lru_put(cache: OrderedDict[tuple, Any], key: tuple, value: Any) -> None:
    """Store *value* under *key* in *cache* with LRU eviction.

    Once the cache exceeds ``_DCT_CACHE_MAXSIZE`` the least-recently-used
    entry is removed.  The new entry is always appended as most-recently-used,
    so eviction (``popitem(last=False)``) removes the oldest surviving entry,
    never the one just inserted.
    """
    cache[key] = value
    if len(cache) > _DCT_CACHE_MAXSIZE:
        cache.popitem(last=False)


def _get_dct_matrix(n: int, device: Any) -> Any:
    """Return a cached (n, n) orthonormal DCT-II matrix on *device*.

    The matrix ``A`` satisfies ``y = A @ x`` for the length-*n* DCT-II
    with ``norm="ortho"``.  It is built in float64 for accuracy and cached
    per ``(n, device)``.  Callers cast to float32 via ``.float()`` as needed.

    Parameters
    ----------
    n : int
        Length of the DCT.
    device : torch.device or str
        Target device.

    Returns
    -------
    torch.Tensor, shape (n, n), dtype float64
        Orthonormal DCT-II matrix on *device*.
    """
    import torch

    key = (n, str(device))
    if key not in _DCT_MATRIX_CACHE:
        k = torch.arange(n, dtype=torch.float64, device=device)
        i = torch.arange(n, dtype=torch.float64, device=device)
        A = torch.cos(math.pi * (i[None, :] + 0.5) * k[:, None] / n)
        A[0] *= 1.0 / math.sqrt(n)
        A[1:] *= math.sqrt(2.0 / n)
        _lru_put(_DCT_MATRIX_CACHE, key, A)
    else:
        _DCT_MATRIX_CACHE.move_to_end(key)
    return _DCT_MATRIX_CACHE[key]


def _get_dct_twiddles(n: int, dtype: Any, device: Any) -> tuple:
    """Return cached (cos_k, sin_k, ortho_scale) for a forward DCT of length *n*."""
    import torch

    key = (n, dtype, str(device))
    if key not in _DCT_TWIDDLE_CACHE:
        k = torch.arange(n, dtype=dtype, device=device)
        theta = math.pi * k / (2.0 * n)
        cos_k = torch.cos(theta)
        sin_k = torch.sin(theta)
        scale = torch.empty(n, dtype=dtype, device=device)
        scale[0] = 1.0 / math.sqrt(n)
        scale[1:] = 1.0 / math.sqrt(n / 2)
        _lru_put(_DCT_TWIDDLE_CACHE, key, (cos_k, sin_k, scale))
    else:
        _DCT_TWIDDLE_CACHE.move_to_end(key)
    return _DCT_TWIDDLE_CACHE[key]


def _get_idct_twiddles(n: int, dtype: Any, device: Any) -> tuple:
    """Return cached (cos_k, sin_k, ortho_scale) for an inverse DCT of length *n*."""
    import torch

    key = (n, dtype, str(device))
    if key not in _IDCT_TWIDDLE_CACHE:
        k = torch.arange(n, dtype=dtype, device=device)
        cos_k = torch.cos(math.pi * k / (2.0 * n))
        sin_k = torch.sin(math.pi * k / (2.0 * n))
        scale = torch.empty(n, dtype=dtype, device=device)
        scale[0] = math.sqrt(n)
        scale[1:] = math.sqrt(n / 2)
        _lru_put(_IDCT_TWIDDLE_CACHE, key, (cos_k, sin_k, scale))
    else:
        _IDCT_TWIDDLE_CACHE.move_to_end(key)
    return _IDCT_TWIDDLE_CACHE[key]


def _torch_dct1d(x: Any, norm: str = "ortho") -> Any:
    """Orthonormal 1-D DCT-II along the last axis.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    norm : str
        Must be ``"ortho"``.

    Returns
    -------
    torch.Tensor
        DCT-II coefficients along the last axis.

    Notes
    -----
    Algorithm reorders the input (even/odd interleaving), applies a real
    FFT, then multiplies by precomputed twiddle factors to obtain the
    DCT-II spectrum.  Twiddle factors are cached per (length, dtype, device)
    to avoid recomputation on every call.
    """
    import torch

    n = x.shape[-1]
    cos_k, sin_k, scale = _get_dct_twiddles(n, x.dtype, x.device)
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    Vc = torch.fft.fft(v, n=n, dim=-1)
    # Re(Vc * exp(-i*theta)) = Vc.real*cos + Vc.imag*sin
    y = Vc.real * cos_k + Vc.imag * sin_k
    if norm == "ortho":
        return y * scale
    return 2.0 * y


def _torch_idct1d(x: Any, norm: str = "ortho") -> Any:
    """Orthonormal 1-D DCT-III (inverse DCT-II) along the last axis.

    Parameters
    ----------
    x : torch.Tensor
        DCT coefficient tensor.
    norm : str
        Must be ``"ortho"``.

    Returns
    -------
    torch.Tensor
        Reconstructed values along the last axis.
    """
    import torch

    n = x.shape[-1]
    cos_k, sin_k, scale = _get_idct_twiddles(n, x.dtype, x.device)
    xn = x * scale if norm == "ortho" else x / 2
    # Anti-Hermitian imaginary part (exploits real-signal symmetry of forward FFT)
    Vt_i = torch.cat([torch.zeros_like(xn[..., :1]), -xn[..., 1:].flip(-1)], dim=-1)
    V_r = xn * cos_k - Vt_i * sin_k
    V_i = xn * sin_k + Vt_i * cos_k
    V = torch.complex(V_r, V_i)
    v = torch.fft.ifft(V, n=n, dim=-1).real
    y = torch.zeros_like(v)
    y[..., ::2] = v[..., : math.ceil(n / 2)]
    y[..., 1::2] = v[..., math.ceil(n / 2) :].flip(-1)
    return y


def _torch_dctn(x: Any, norm: str = "ortho") -> Any:
    """N-dimensional orthonormal DCT-II (applied to each axis sequentially).

    For the common 2-D case the two passes are fused into a single
    transpose-free contiguous batch to reduce kernel launch overhead.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of arbitrary rank.
    norm : str
        Must be ``"ortho"``.

    Returns
    -------
    torch.Tensor
        DCT-II coefficients, same shape as *x*.
    """
    if x.ndim == 2:
        # Axis 1 (last): DCT along rows.
        y = _torch_dct1d(x, norm=norm)
        # Axis 0: transpose so axis 0 becomes the last axis, apply DCT, transpose back.
        return _torch_dct1d(y.t().contiguous(), norm=norm).t().contiguous()
    y = x
    for i in range(y.ndim):
        y = _torch_dct1d(y.transpose(-1, i).contiguous(), norm=norm).transpose(-1, i)
    return y


def _torch_idctn(x: Any, norm: str = "ortho") -> Any:
    """N-dimensional orthonormal inverse DCT-II (DCT-III).

    For the common 2-D case the two passes are fused into a single
    transpose-free contiguous batch.

    Parameters
    ----------
    x : torch.Tensor
        DCT coefficient tensor of arbitrary rank.
    norm : str
        Must be ``"ortho"``.

    Returns
    -------
    torch.Tensor
        Reconstructed tensor, same shape as *x*.
    """
    if x.ndim == 2:
        y = _torch_idct1d(x, norm=norm)
        return _torch_idct1d(y.t().contiguous(), norm=norm).t().contiguous()
    y = x
    for i in range(y.ndim):
        y = _torch_idct1d(y.transpose(-1, i).contiguous(), norm=norm).transpose(-1, i)
    return y
