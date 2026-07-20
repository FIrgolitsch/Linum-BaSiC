"""linum-basic — illumination correction for optical microscopy images.

This package is built up across a series of incremental pull requests. At this
stage only the array backend abstraction (:mod:`linum_basic.backend`), the
tile-mosaic model (:mod:`linum_basic.mosaic`), the Torch compile-cache helpers
(:mod:`linum_basic._torch_cache`), and the parallelism layer
(:mod:`linum_basic._parallel`) are available. The public estimator API is
added by later pull requests in the chain.
"""

__version__ = "2.0.0"
