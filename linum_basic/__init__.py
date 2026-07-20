"""linum-basic — illumination correction for optical microscopy images.

This package is built up across a series of incremental pull requests. At this
stage the available public surface is the BaSiC flat-field / dark-field
estimator (:class:`linum_basic.core.BaSiC`) and the low-level ALM solver and
soft-threshold operator (:mod:`linum_basic.algorithms`). The array backend
abstraction (:mod:`linum_basic.backend`), the tile-mosaic model
(:mod:`linum_basic.mosaic`), the Torch compile-cache helpers
(:mod:`linum_basic._torch_cache`), and the parallelism layer
(:mod:`linum_basic._parallel`) are also importable. The mosaic fitter, tuning
API, I/O helpers, and one-call convenience wrapper are added by later pull
requests in the chain.

Public API
----------
BaSiC
    Main estimator class.  See :class:`linum_basic.core.BaSiC` for full
    documentation.
inexact_alm_l1, inexact_alm_l1_batched, shrink
    Low-level ALM solver and soft-threshold operator exposed via
    :mod:`linum_basic.algorithms` for users who want direct access to the
    numerical core (e.g. custom reweighting schemes).

Examples
--------
>>> import numpy as np
>>> from linum_basic import BaSiC
>>> stack = np.random.rand(30, 64, 64).astype("float32")
>>> model = BaSiC(stack)
>>> _ = model.run()  # prepare() is called automatically
>>> flatfield = model.get_flatfield()
"""

from linum_basic.algorithms import inexact_alm_l1, inexact_alm_l1_batched, shrink
from linum_basic.core import BaSiC

__version__ = "2.0.0"
__all__ = [
    "BaSiC",
    "__version__",
    "inexact_alm_l1",
    "inexact_alm_l1_batched",
    "shrink",
]
