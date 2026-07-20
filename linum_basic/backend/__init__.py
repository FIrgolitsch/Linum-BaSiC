"""Array-backend abstraction for linum-basic.

Provides a unified namespace for NumPy and PyTorch operations, allowing
the core ALM optimisation loop to run on CPU (NumPy) or GPU (PyTorch)
without code duplication.

This is the package's public entry point — the ``Backend`` enum, the
:class:`ArrayNamespace` facade, and the :func:`get_xp` factory all live
here.  The module-level DCT / cache infrastructure that historically sat
below this docstring has been physically split across two sibling modules
per D026/D029 so the port-vs-capability boundary is reviewable in the S05
PR chain:

- :mod:`linum_basic.backend._torch` — Torch-only "new capability" code
  (the three LRU-bounded DCT caches, twiddle-factor and orthonormal-matrix
  builders, FFT-based DCT-II/DCT-III, DCT-kernel-mode selector, and cache
  helpers).  Every symbol is re-exported below so existing
  ``from linum_basic.backend import _get_dct_matrix`` /
  ``from linum_basic.backend import _DCT_MATRIX_CACHE`` call sites continue
  to resolve, and so cache **object identity** is preserved: the bindings
  in this package namespace point at the same ``OrderedDict`` instances
  that ``_torch._get_dct_matrix`` mutates, so
  ``backend.clear_dct_caches()`` and direct ``_DCT_MATRIX_CACHE`` access
  observe the same mutation (required by ``tests/test_dct_cache_bounds.py``).
- :mod:`linum_basic.backend._numpy` — thin re-export of the SciPy
  orthonormal DCT-II / DCT-III primitives that mirror the upstream
  ``pybasic/shading_correction.py`` port surface.

The split is a pure file-move with no logic changes, so the darkfield
regression and vignette validation evidence (K02) collected against the
pre-split ``backend.py`` carries through bit-identically (re-proven in S04
T03).
"""

from __future__ import annotations

import enum
from typing import Any

import numpy as np

# Redundant-as aliases mark every symbol below as an *intentional re-export*
# (the canonical Python idiom that ruff F401 recognises).  ``backend`` must
# continue to expose these names directly so existing
# ``from linum_basic.backend import _DCT_MATRIX_CACHE`` /
# ``backend_mod._get_dct_matrix`` call sites resolve and so cache object
# identity is preserved (tests/test_dct_cache_bounds.py).
from linum_basic.backend._numpy import scipy_dctn as scipy_dctn
from linum_basic.backend._numpy import scipy_idctn as scipy_idctn
from linum_basic.backend._torch import _DCT_CACHE_MAXSIZE as _DCT_CACHE_MAXSIZE
from linum_basic.backend._torch import _DCT_MATRIX_CACHE as _DCT_MATRIX_CACHE
from linum_basic.backend._torch import _DCT_TWIDDLE_CACHE as _DCT_TWIDDLE_CACHE
from linum_basic.backend._torch import _IDCT_TWIDDLE_CACHE as _IDCT_TWIDDLE_CACHE
from linum_basic.backend._torch import _get_dct_matrix as _get_dct_matrix
from linum_basic.backend._torch import _get_dct_twiddles as _get_dct_twiddles
from linum_basic.backend._torch import _get_idct_twiddles as _get_idct_twiddles
from linum_basic.backend._torch import _lru_put as _lru_put
from linum_basic.backend._torch import _torch_dct1d as _torch_dct1d
from linum_basic.backend._torch import _torch_dctn as _torch_dctn
from linum_basic.backend._torch import _torch_idct1d as _torch_idct1d
from linum_basic.backend._torch import _torch_idctn as _torch_idctn
from linum_basic.backend._torch import clear_dct_caches as clear_dct_caches
from linum_basic.backend._torch import read_dct_kernel_mode as read_dct_kernel_mode

__all__ = ["ArrayNamespace", "Backend", "clear_dct_caches", "get_xp"]


class Backend(enum.StrEnum):
    """Supported compute backends.

    Attributes
    ----------
    NUMPY : str
        Pure NumPy / SciPy on CPU.
    TORCH : str
        PyTorch — CPU or CUDA GPU; MPS is not supported.
    """

    NUMPY = "numpy"
    TORCH = "torch"


class ArrayNamespace:
    """Thin wrapper providing a common API over NumPy and PyTorch.

    Parameters
    ----------
    backend : Backend
        Which backend to use.
    device : str or None
        PyTorch device string (e.g. ``"cuda"``, ``"cpu"``).
        Ignored when *backend* is ``Backend.NUMPY``.

    Raises
    ------
    ImportError
        If ``Backend.TORCH`` is requested but PyTorch is not installed.
    ValueError
        If an unsupported *backend* value is given.
    """

    def __init__(self, backend: Backend, device: str | None = None) -> None:
        self._backend = backend
        self._torch: Any = None
        self._device: Any = None
        if backend is Backend.TORCH:
            try:
                import torch
            except ImportError as exc:
                msg = "PyTorch is required for the 'torch' backend. Install it with: uv sync --extra gpu"
                raise ImportError(msg) from exc
            self._torch = torch
            self._device = torch.device(device or "cpu")
        elif backend is Backend.NUMPY:
            pass
        else:
            msg = f"Unknown backend: {backend!r}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Array creation
    # ------------------------------------------------------------------

    def asarray(self, x: np.ndarray, dtype: type | None = None) -> Any:
        """Convert a NumPy array to the backend's native type.

        Parameters
        ----------
        x : numpy.ndarray
            Source array.
        dtype : type or None
            Target dtype.  ``None`` preserves the source dtype mapped to the
            closest backend equivalent.

        Returns
        -------
        object
            Native array on the configured device.
        """
        if self._backend is Backend.NUMPY:
            return np.asarray(x, dtype=dtype)
        # Pass-through: already a tensor on the right device with no dtype coercion needed
        if isinstance(x, self._torch.Tensor) and x.device == self._device and dtype is None:
            return x
        # For dtype resolution, always go through numpy to get a numpy dtype
        np_dtype = np.dtype(dtype) if dtype is not None else np.asarray(x).dtype
        torch_dtype = self._numpy_dtype_to_torch(np_dtype)
        return self._torch.as_tensor(np.asarray(x), dtype=torch_dtype, device=self._device)

    def to_numpy(self, x: Any) -> np.ndarray:
        """Convert a backend array back to a NumPy array.

        Parameters
        ----------
        x : object
            Backend-native array.

        Returns
        -------
        numpy.ndarray
            CPU NumPy copy.
        """
        if self._backend is Backend.NUMPY:
            if isinstance(x, np.ndarray):
                return x
            return np.asarray(x)
        return x.detach().cpu().numpy()

    def astype(self, x: Any, dtype: type) -> Any:
        """Cast a backend array to *dtype* without a device round-trip.

        Parameters
        ----------
        x : object
            Backend-native array.
        dtype : type
            Target NumPy dtype (e.g. ``np.float32``, ``np.float64``).

        Returns
        -------
        object
            Array with the requested element type, on the same device as *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.asarray(x, dtype=dtype)
        return x.to(self._numpy_dtype_to_torch(np.dtype(dtype)))

    def zeros(self, shape: tuple[int, ...], dtype: type | None = None) -> Any:
        """Return a zero-filled array of the given shape.

        Parameters
        ----------
        shape : tuple of int
            Output shape.
        dtype : type or None
            Element type.

        Returns
        -------
        object
            Backend-native zero array.
        """
        if self._backend is Backend.NUMPY:
            return np.zeros(shape, dtype=dtype)
        torch_dtype = self._numpy_dtype_to_torch(np.dtype(dtype or np.float32))
        return self._torch.zeros(shape, dtype=torch_dtype, device=self._device)

    def ones(self, shape: tuple[int, ...], dtype: type | None = None) -> Any:
        """Return a ones-filled array of the given shape.

        Parameters
        ----------
        shape : tuple of int
            Output shape.
        dtype : type or None
            Element type.

        Returns
        -------
        object
            Backend-native ones array.
        """
        if self._backend is Backend.NUMPY:
            return np.ones(shape, dtype=dtype)
        torch_dtype = self._numpy_dtype_to_torch(np.dtype(dtype or np.float32))
        return self._torch.ones(shape, dtype=torch_dtype, device=self._device)

    def zeros_like(self, x: Any) -> Any:
        """Return a zero array matching the shape and dtype of *x*.

        Parameters
        ----------
        x : Any
            Reference array.

        Returns
        -------
        Any
            Zero array matching *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.zeros_like(x)
        return self._torch.zeros_like(x)

    def ones_like(self, x: Any) -> Any:
        """Return a ones array matching the shape and dtype of *x*.

        Parameters
        ----------
        x : Any
            Reference array.

        Returns
        -------
        Any
            Ones array matching *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.ones_like(x)
        return self._torch.ones_like(x)

    # ------------------------------------------------------------------
    # Element-wise math
    # ------------------------------------------------------------------

    def abs(self, x: Any) -> Any:
        """Element-wise absolute value.

        Parameters
        ----------
        x : Any
            Input array.

        Returns
        -------
        Any
            ``|x|``.
        """
        if self._backend is Backend.NUMPY:
            return np.abs(x)
        return self._torch.abs(x)

    def sign(self, x: Any) -> Any:
        """Element-wise sign (-1, 0, or 1).

        Parameters
        ----------
        x : Any
            Input array.

        Returns
        -------
        Any
            Sign of *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.sign(x)
        return self._torch.sign(x)

    def maximum(self, x: Any, y: Any) -> Any:
        """Element-wise maximum of *x* and *y*.

        Parameters
        ----------
        x : Any
            First operand.
        y : Any
            Second operand (may be a scalar).

        Returns
        -------
        Any
            ``max(x, y)`` element-wise.
        """
        if self._backend is Backend.NUMPY:
            return np.maximum(x, y)
        if isinstance(y, (int, float)):
            return self._torch.clamp_min(x, y)
        return self._torch.maximum(x, y)

    def minimum(self, x: Any, y: Any) -> Any:
        """Element-wise minimum of *x* and *y*.

        Parameters
        ----------
        x : Any
            First operand.
        y : Any
            Second operand (may be a scalar).

        Returns
        -------
        Any
            ``min(x, y)`` element-wise.
        """
        if self._backend is Backend.NUMPY:
            return np.minimum(x, y)
        if isinstance(y, (int, float)):
            return self._torch.clamp_max(x, y)
        return self._torch.minimum(x, y)

    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Element-wise selection from *x* or *y* depending on *condition*.

        Parameters
        ----------
        condition : Any
            Boolean array or scalar.
        x : Any
            Values used where *condition* is ``True``.
        y : Any
            Values used where *condition* is ``False``.

        Returns
        -------
        Any
            Array of selected values on the same device as the inputs.
        """
        if self._backend is Backend.NUMPY:
            return np.where(condition, x, y)
        return self._torch.where(condition, x, y)

    def clone(self, x: Any) -> Any:
        """Return an independent copy of *x*, stripping view metadata.

        For the NumPy backend this is a no-op since arithmetic operations
        already produce new arrays.  For the Torch backend, ``x.clone()``
        strips the ``ADInplaceOrView`` dispatch key that ``reshape`` and
        same-dtype ``.to()`` calls attach to their outputs, preventing
        spurious ``torch._dynamo`` guard failures when the tensor is fed back
        as a compiled-step input on the next iteration.

        Parameters
        ----------
        x : Any
            Array or tensor to clone.

        Returns
        -------
        Any
            An independent tensor with the same values and device as *x*.
        """
        if self._backend is Backend.NUMPY:
            return x
        return x.clone()

    def copysign(self, magnitude: Any, sign_source: Any) -> Any:
        """Element-wise copysign: return *magnitude* with the sign of *sign_source*.

        Parameters
        ----------
        magnitude : Any
            Array of non-negative magnitudes.
        sign_source : Any
            Array from which the sign information is taken.

        Returns
        -------
        Any
            Array with values from *magnitude* and signs from *sign_source*.
        """
        if self._backend is Backend.NUMPY:
            return np.copysign(magnitude, sign_source)
        return self._torch.copysign(magnitude, sign_source)

    def mean(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Compute the arithmetic mean along an axis.

        Parameters
        ----------
        x : Any
            Input array or tensor.
        axis : int or None
            Axis to reduce along.  ``None`` reduces all elements to a scalar.
        keepdims : bool
            If ``True`` the reduced axis is kept as a dimension of size 1.

        Returns
        -------
        Any
            Reduced result on the same backend as *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.mean(x, axis=axis, keepdims=keepdims)
        if axis is None:
            return self._torch.mean(x)
        return self._torch.mean(x, dim=axis, keepdim=keepdims)

    def sum(self, x: Any, axis: int | None = None, keepdims: bool = False) -> Any:
        """Compute the sum of array elements along an optional axis.

        Parameters
        ----------
        x : object
            Backend-native array.
        axis : int or None
            Axis along which to reduce.  If ``None``, reduces over all elements.
        keepdims : bool
            If ``True``, the reduced axis is kept as a size-1 dimension.

        Returns
        -------
        object
            Reduced array (or scalar when *axis* is ``None``).
        """
        if self._backend is Backend.NUMPY:
            return np.sum(x, axis=axis, keepdims=keepdims)
        if axis is None:
            return self._torch.sum(x)
        return self._torch.sum(x, dim=axis, keepdim=keepdims)

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------

    def norm_fro(self, x: Any) -> float:
        """Compute the Frobenius norm of *x*.

        Parameters
        ----------
        x : Any
            2-D (or flat) array.

        Returns
        -------
        float
            Frobenius norm ``‖x‖_F``.
        """
        if self._backend is Backend.NUMPY:
            return float(np.linalg.norm(x, "fro"))
        return float(self._torch.linalg.norm(x, ord="fro"))

    def norm_fro_batched(self, x: Any) -> Any:
        """Per-batch Frobenius norm along trailing matrix dimensions.

        Parameters
        ----------
        x : Any
            Array with shape ``(Z, …)`` where the last two dimensions form
            the matrix whose norm is computed for each leading batch index.

        Returns
        -------
        Any
            Shape ``(Z,)`` on the active device (Torch) or NumPy array.
        """
        if self._backend is Backend.NUMPY:
            flat = x.reshape(x.shape[0], -1)
            return np.linalg.norm(flat, axis=1)
        return self._torch.linalg.vector_norm(x.reshape(x.shape[0], -1), ord=2, dim=1)

    def min(self, x: Any) -> float:
        """Return the global minimum value of *x* as a Python float.

        Avoids materialising the full tensor to CPU when on GPU backends.

        Parameters
        ----------
        x : object
            Backend-native array.

        Returns
        -------
        float
            Minimum element.
        """
        if self._backend is Backend.NUMPY:
            return float(np.min(x))
        return float(self._torch.min(x))

    def min_along(self, x: Any, axis: int, *, keepdims: bool = False) -> Any:
        """Minimum along *axis*, keeping result on the active device.

        Returns
        -------
        Any
            Reduced array on the same backend as *x*.
        """
        if self._backend is Backend.NUMPY:
            return np.min(x, axis=axis, keepdims=keepdims)
        return self._torch.amin(x, dim=axis, keepdim=keepdims)

    def svd_leading_singular(self, x: Any, *, n_iter: int = 30) -> float:
        """Return the largest singular value of *x*.

        Uses power iteration on ``x @ x.T`` for GPU backends to avoid a full
        SVD and the associated device synchronisation.  NumPy falls back to
        ``numpy.linalg.svd(compute_uv=False)``.

        Parameters
        ----------
        x : object
            2-D input matrix.
        n_iter : int
            Power-iteration steps (GPU path only).

        Returns
        -------
        float
            Largest singular value σ₁.
        """
        if self._backend is Backend.NUMPY:
            return float(np.linalg.svd(x, compute_uv=False)[0])
        return float(self._svd_leading_singular_torch(x, n_iter=n_iter))

    def svd_leading_singular_batched(self, x: Any, *, n_iter: int = 30) -> Any:
        """Return the largest singular value for each batch matrix.

        Parameters
        ----------
        x : object
            3-D array ``(Z, n, m)``.
        n_iter : int
            Power-iteration steps.

        Returns
        -------
        object
            Shape ``(Z,)`` tensor/array on the active device.
        """
        if self._backend is Backend.NUMPY:
            return np.array([float(np.linalg.svd(x[z], compute_uv=False)[0]) for z in range(x.shape[0])])
        return self._svd_leading_singular_torch_batched(x, n_iter=n_iter)

    def _svd_leading_singular_torch(self, x: Any, *, n_iter: int) -> Any:
        """Power iteration for σ₁ of a single 2-D matrix (returns Python float)."""
        sigma = self._svd_leading_singular_torch_batched(x.unsqueeze(0), n_iter=n_iter)
        return float(sigma[0])

    def _svd_leading_singular_torch_batched(self, x: Any, *, n_iter: int) -> Any:
        """Power iteration for σ₁ of batched matrices ``(Z, n, m)``.

        Uses a deterministic uniform initial vector so the estimate is
        reproducible across processes and GPUs (the leading right-singular
        vector of sorted, mostly-positive image data is close to uniform,
        so this also converges quickly).
        """
        torch = self._torch
        z, _n, m = x.shape
        v = torch.ones(z, m, 1, dtype=x.dtype, device=x.device)
        v = v / (torch.linalg.vector_norm(v, dim=1, keepdim=True) + 1e-9)
        for _ in range(n_iter):
            u = torch.bmm(x, v)
            u = u / (torch.linalg.vector_norm(u, dim=1, keepdim=True) + 1e-9)
            v = torch.bmm(x.transpose(1, 2), u)
            v = v / (torch.linalg.vector_norm(v, dim=1, keepdim=True) + 1e-9)
        return torch.linalg.vector_norm(torch.bmm(x, v), dim=(1, 2))

    def inference_mode(self) -> Any:
        """Return a context manager that disables gradient tracking.

        On the Torch backend this returns ``torch.inference_mode()``.
        On the NumPy backend it returns a no-op context manager so the
        ALM loop can unconditionally use ``with xp.inference_mode()``.

        Returns
        -------
        contextlib.AbstractContextManager
            Context manager that disables gradient computation (Torch) or
            is a no-op (NumPy).
        """
        if self._backend is Backend.TORCH:
            return self._torch.inference_mode()
        import contextlib

        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # DCT
    # ------------------------------------------------------------------

    def dctn(self, x: Any, norm: str = "ortho") -> Any:
        """N-dimensional orthonormal DCT-II.

        Parameters
        ----------
        x : object
            Input array.
        norm : str
            Normalisation mode.  Only ``"ortho"`` is supported.

        Returns
        -------
        object
            DCT-II coefficients with the same shape as *x*.

        Notes
        -----
        The Torch backend uses an FFT-based implementation that matches
        SciPy's ``scipy.fft.dctn`` output to within floating-point precision.
        """
        if self._backend is Backend.NUMPY or isinstance(x, np.ndarray):
            return scipy_dctn(x, norm=norm)
        return _torch_dctn(x, norm=norm)

    def idctn(self, x: Any, norm: str = "ortho") -> Any:
        """N-dimensional orthonormal inverse DCT-II (DCT-III).

        Parameters
        ----------
        x : object
            DCT coefficient array.
        norm : str
            Normalisation mode.  Only ``"ortho"`` is supported.

        Returns
        -------
        object
            Reconstructed array with the same shape as *x*.
        """
        if self._backend is Backend.NUMPY or isinstance(x, np.ndarray):
            return scipy_idctn(x, norm=norm)
        return _torch_idctn(x, norm=norm)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _numpy_dtype_to_torch(self, dtype: np.dtype) -> object:
        """Map a NumPy dtype to the corresponding PyTorch dtype.

        Parameters
        ----------
        dtype : numpy.dtype
            Source dtype.

        Returns
        -------
        torch.dtype
            Closest PyTorch equivalent.  Defaults to ``float32`` for unknown
            types.

        Notes
        -----
        Supported mappings:

        ============== ====================
        NumPy dtype    PyTorch dtype
        ============== ====================
        ``float32``    ``torch.float32``
        ``float64``    ``torch.float64``
        ``int32``      ``torch.int32``
        ``int64``      ``torch.int64``
        ``bool_``      ``torch.bool``
        *other*        ``torch.float32``
        ============== ====================
        """
        _map = {
            np.float32: self._torch.float32,
            np.float64: self._torch.float64,
            np.int32: self._torch.int32,
            np.int64: self._torch.int64,
            np.bool_: self._torch.bool,
        }
        return _map.get(dtype.type, self._torch.float32)


def get_xp(backend: str | Backend, device: str | None = None) -> ArrayNamespace:
    """Construct an :class:`ArrayNamespace` for the requested backend.

    Parameters
    ----------
    backend : str or Backend
        ``"numpy"``, ``"torch"``, or ``"auto"``.  ``"auto"`` selects the
        Torch backend with CUDA when available, otherwise falls back
        to NumPy.  MPS is not supported and raises
        :exc:`NotImplementedError` when requested explicitly.
    device : str or None
        PyTorch device string (e.g. ``"cuda:0"``).  Ignored for NumPy.

    Returns
    -------
    ArrayNamespace
        Configured array namespace ready for use in the ALM loop.

    Examples
    --------
    >>> xp = get_xp("numpy")
    >>> arr = xp.zeros((4, 4))

    >>> xp_gpu = get_xp("auto")  # picks GPU if available
    """
    if isinstance(backend, str) and backend == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return ArrayNamespace(Backend.TORCH, device or "cuda")
            # MPS is intentionally excluded: svdvals and float64 are not supported.
        except ImportError:
            pass
        return ArrayNamespace(Backend.NUMPY)

    b = Backend(backend) if isinstance(backend, str) else backend
    if b is Backend.TORCH:
        resolved_device = device or "cpu"
        if resolved_device.startswith("mps"):
            msg = (
                "MPS is not supported as a backend for linum-basic: PyTorch MPS lacks "
                "float64 and svdvals, both required by the ALM solver. "
                "Use backend='torch' with device='cuda' or backend='numpy' instead."
            )
            raise NotImplementedError(msg)
    return ArrayNamespace(b, device)
