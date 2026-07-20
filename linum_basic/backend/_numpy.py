"""NumPy/SciPy DCT-II / DCT-III re-export (port-relevant path).

Thin re-export of the SciPy orthonormal DCT-II / DCT-III primitives used by
the NumPy branches of :meth:`linum_basic.backend.ArrayNamespace.dctn` and
:meth:`linum_basic.backend.ArrayNamespace.idctn`.  Per D029, this is the
"port-relevant" DCT surface: it mirrors the DCT calls in the upstream
401-line ``pybasic/shading_correction.py`` (which calls
``scipy.fft.dctn`` / ``scipy.fft.idctn`` directly), whereas the Torch-only
FFT-based DCT implementation in :mod:`linum_basic.backend._torch` is new
functionality with no upstream equivalent.

Centralising the SciPy import here gives the future S05 port-vs-capability
PR chain a single file that represents the "NumPy port" half of the backend
DCT boundary, complementing the Torch "new capability" half in
:mod:`linum_basic.backend._torch`.

This file is a verbatim re-export of the two symbols previously imported at
module scope in the pre-split ``backend.py`` — no logic changes.
"""

from __future__ import annotations

from scipy.fft import dctn as scipy_dctn
from scipy.fft import idctn as scipy_idctn

__all__ = ["scipy_dctn", "scipy_idctn"]
