"""Tests for the device-resident ALM solver and the fit path built on it.

The device solver must give the same fields as the generic step it replaces
on the Torch backend.  The comparisons run on Torch-CPU so they need no GPU;
tests marked ``cuda`` repeat the key ones on a GPU when one is present.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from linum_basic._alm import inexact_alm_l1
from linum_basic._alm_device import AlmProblem, read_alm_solver
from linum_basic._parallel import thread_map_cuda_devices
from linum_basic.backend import Backend, get_xp
from linum_basic.core import BaSiC
from linum_basic.fit import MosaicFit, apply_fit
from linum_basic.mosaic import MosaicGrid

torch = pytest.importorskip("torch", reason="PyTorch not installed")

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _stack(n: int = 24, size: int = 32, *, offset: float = 0.0, seed: int = 0) -> np.ndarray:
    """Images with a smooth vignette, a per-image level and sparse bright spots."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size] / (size - 1)
    flat = 1.0 - 0.5 * ((xx - 0.45) ** 2 + (yy - 0.55) ** 2)
    level = rng.uniform(20.0, 120.0, size=(n, 1, 1))
    stack = level * flat[None] * rng.uniform(0.97, 1.03, size=(n, size, size)) + offset
    spots = rng.random((n, size, size)) < 0.03
    stack[spots] *= 3.0
    return stack.astype(np.float32)


def _legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Select the generic step, eager, for the reference run."""
    monkeypatch.setenv("LINUM_BASIC_ALM_SOLVER", "legacy")
    monkeypatch.setenv("LINUM_BASIC_ALM_COMPILE_MODE", "off")


def _fit(stack: np.ndarray, *, device: str = "cpu", darkfield: bool = False, warm: bool = True) -> BaSiC:
    model = BaSiC(stack, backend="torch", device=device, estimate_darkfield=darkfield)
    model.working_size = stack.shape[1]
    model.max_reweighting_iterations = 6
    model.warm_start_reweighting = warm
    model.run()
    return model


class TestSolverSelection:
    def test_device_is_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LINUM_BASIC_ALM_SOLVER", raising=False)
        assert read_alm_solver() == "device"

    def test_legacy_switch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LINUM_BASIC_ALM_SOLVER", " Legacy ")
        assert read_alm_solver() == "legacy"

    def test_basic_uses_the_device_problem_on_torch_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LINUM_BASIC_ALM_SOLVER", raising=False)
        stack = _stack(8, 16)
        on_torch = BaSiC(stack, backend="torch", device="cpu")
        on_torch.working_size = 16
        on_torch.prepare()
        assert on_torch._problem is not None
        on_numpy = BaSiC(stack, backend="numpy")
        on_numpy.working_size = 16
        on_numpy.prepare()
        assert on_numpy._problem is None

    def test_legacy_switch_keeps_the_host_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _legacy(monkeypatch)
        model = BaSiC(_stack(8, 16), backend="torch", device="cpu")
        model.working_size = 16
        model.prepare()
        assert model._problem is None


class TestParityWithGenericStep:
    """Same fields, same pass and iteration counts as the generic step."""

    @pytest.mark.parametrize("warm", [True, False])
    def test_flatfield(self, monkeypatch: pytest.MonkeyPatch, warm: bool) -> None:
        stack = _stack()
        new = _fit(stack, warm=warm)
        _legacy(monkeypatch)
        old = _fit(stack, warm=warm)
        assert new.reweighting_iteration == old.reweighting_iteration
        assert new.last_alm_iterations == old.last_alm_iterations
        np.testing.assert_allclose(new.get_flatfield(), old.get_flatfield(), atol=2e-4)
        assert np.count_nonzero(new.get_darkfield()) == 0

    def test_darkfield(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stack = _stack(offset=8.0, seed=3)
        new = _fit(stack, darkfield=True)
        _legacy(monkeypatch)
        old = _fit(stack, darkfield=True)
        assert new.reweighting_iteration == old.reweighting_iteration
        np.testing.assert_allclose(new.get_flatfield(), old.get_flatfield(), atol=1e-3)
        scale = max(float(np.abs(old.get_darkfield()).max()), 1.0)
        np.testing.assert_allclose(new.get_darkfield(), old.get_darkfield(), atol=1e-3 * scale)

    def test_sorted_stack_is_available_on_the_host(self) -> None:
        stack = _stack(10, 16)
        model = BaSiC(stack, backend="torch", device="cpu")
        model.working_size = 16
        model.prepare()
        np.testing.assert_array_equal(model.img_sort, np.sort(model.img_stack_resized, axis=0))

    @requires_cuda
    def test_flatfield_on_cuda_matches_cpu(self) -> None:
        stack = _stack(64, 48)
        on_gpu = _fit(stack, device="cuda:0")
        on_cpu = _fit(stack, device="cpu")
        assert on_gpu.reweighting_iteration == on_cpu.reweighting_iteration
        np.testing.assert_allclose(on_gpu.get_flatfield(), on_cpu.get_flatfield(), atol=2e-4)


class TestFunctionApi:
    """``inexact_alm_l1`` keeps its outputs when routed through the device solver."""

    def _both(self, monkeypatch: pytest.MonkeyPatch, **kwargs: object) -> tuple[tuple, tuple]:
        stack = np.sort(_stack(16, 24), axis=0)
        xp = get_xp(Backend.TORCH, "cpu")
        new = inexact_alm_l1(stack, 0.5, 0.2, xp=xp, return_state=True, **kwargs)
        _legacy(monkeypatch)
        old = inexact_alm_l1(stack, 0.5, 0.2, xp=xp, return_state=True, **kwargs)
        return new, old

    @pytest.mark.parametrize("darkfield", [False, True])
    def test_outputs_match(self, monkeypatch: pytest.MonkeyPatch, darkfield: bool) -> None:
        new, old = self._both(monkeypatch, estimate_darkfield=darkfield)
        scale = float(np.abs(old[0]).max())
        for got, want in zip(new[:3], old[:3], strict=True):
            assert got.shape == want.shape
            np.testing.assert_allclose(got, want, atol=2e-4 * scale)
        assert new[3]["alm_iterations"] == old[3]["alm_iterations"]
        assert set(new[3]) == set(old[3])
        for key in ("Sf", "B", "D_field"):
            np.testing.assert_allclose(new[3][key], old[3][key], atol=2e-4 * scale)

    def test_weights_and_warm_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stack = np.sort(_stack(16, 24), axis=0)
        rng = np.random.default_rng(1)
        weight = rng.uniform(0.5, 2.0, size=stack.shape).astype(np.float32)
        xp = get_xp(Backend.TORCH, "cpu")
        first = inexact_alm_l1(stack, 0.5, 0.2, xp=xp, estimate_darkfield=False, return_state=True)
        new = inexact_alm_l1(stack, 0.5, 0.2, xp=xp, estimate_darkfield=False, weight=weight, warm_start=first[3])
        _legacy(monkeypatch)
        old = inexact_alm_l1(stack, 0.5, 0.2, xp=xp, estimate_darkfield=False, weight=weight, warm_start=first[3])
        scale = float(np.abs(old[0]).max())
        np.testing.assert_allclose(new[0], old[0], atol=2e-4 * scale)
        np.testing.assert_allclose(new[1], old[1], atol=2e-4 * scale)

    def test_zero_iterations_returns_zeros(self) -> None:
        stack = _stack(6, 16)
        xp = get_xp(Backend.TORCH, "cpu")
        with pytest.warns(UserWarning, match="Maximum ALM iterations"):
            ib, ir, d_field, _ = inexact_alm_l1(stack, 0.5, 0.2, xp=xp, max_iter=0, estimate_darkfield=False)
        assert not ib.any() and not ir.any() and not d_field.any()


class TestAlmProblem:
    def test_residual_and_low_rank_part_add_up_to_the_data(self) -> None:
        stack = np.sort(_stack(16, 24), axis=0)
        xp = get_xp(Backend.TORCH, "cpu")
        problem = AlmProblem.from_numpy(xp, stack)
        result = problem.solve(0.5, 0.2, estimate_darkfield=False)
        residual = xp.to_numpy(problem.residual(result)).copy()
        low_rank = xp.to_numpy(result.ib())
        data = stack.reshape(stack.shape[0], -1)
        assert np.abs(data - low_rank - residual).max() < 1e-3 * float(data.max())

    def test_reweight_gives_mean_one_and_downweights_outliers(self) -> None:
        stack = np.sort(_stack(16, 24), axis=0)
        xp = get_xp(Backend.TORCH, "cpu")
        problem = AlmProblem.from_numpy(xp, stack)
        result = problem.solve(0.5, 0.2, estimate_darkfield=False)
        residual = np.abs(xp.to_numpy(problem.residual(result)).copy())
        problem.reweight(result, float(result.ib_mean_image().mean()), 0.1)
        weights = xp.to_numpy(problem.W)
        assert weights.mean() == pytest.approx(1.0, rel=1e-4)
        assert weights[residual > np.percentile(residual, 99)].mean() < weights[residual <= np.median(residual)].mean()

    def test_sort_on_device_matches_numpy(self) -> None:
        stack = _stack(12, 16)
        xp = get_xp(Backend.TORCH, "cpu")
        problem = AlmProblem.from_numpy(xp, stack, sort=True)
        np.testing.assert_array_equal(problem.sorted_stack_numpy(), np.sort(stack, axis=0))


class TestThreadMap:
    def test_keeps_order_and_uses_every_device(self) -> None:
        seen: set[str] = set()
        gate = threading.Barrier(2, timeout=5)

        def fn(item: int, device: str) -> int:
            seen.add(device)
            if item < 2:
                gate.wait()  # both threads must be inside fn at once
            return item * item

        out = thread_map_cuda_devices(fn, list(range(9)), ["dev:0", "dev:1"])
        assert out == [i * i for i in range(9)]
        assert seen == {"dev:0", "dev:1"}

    def test_empty_input(self) -> None:
        assert thread_map_cuda_devices(lambda item, device: item, [], ["dev:0"]) == []

    def test_raises_the_worker_error(self) -> None:
        def fn(item: int, device: str) -> int:
            if item == 3:
                msg = "plane 3 failed"
                raise ValueError(msg)
            return item

        with pytest.raises(ValueError, match="plane 3 failed"):
            thread_map_cuda_devices(fn, list(range(8)), ["dev:0", "dev:1"])


class TestApplyFit:
    @pytest.mark.parametrize("with_darkfield", [False, True])
    def test_matches_the_formula(self, with_darkfield: bool) -> None:
        rng = np.random.default_rng(5)
        th, tw, nrows, ncols, nz = 6, 5, 3, 4, 4
        volume = rng.uniform(1.0, 50.0, size=(nz, nrows * th, ncols * tw)).astype(np.float32)
        mosaic = MosaicGrid(array=volume, tile_shape=(th, tw))
        flat = rng.uniform(0.6, 1.4, size=(2, th, tw)).astype(np.float32)
        dark = rng.uniform(0.0, 2.0, size=(2, th, tw)).astype(np.float32) if with_darkfield else np.zeros_like(flat)
        fit = MosaicFit(flatfields=flat, darkfields=dark, field_mode="per-z", z_indices=[0, 3])
        before = volume.copy()
        out = apply_fit(mosaic, fit)
        np.testing.assert_array_equal(volume, before)  # the input is not modified
        for z in range(nz):
            pos = 0 if z < 2 else 1
            tiles = before[z].reshape(nrows, th, ncols, tw)
            want = (tiles - dark[pos][None, :, None, :]) / (flat[pos][None, :, None, :] + 1e-6)
            np.testing.assert_array_equal(out[z].reshape(nrows, th, ncols, tw), want.astype(np.float32))

    def test_extra_rows_are_filled_from_the_first_valid_row(self) -> None:
        th, tw = 6, 4
        volume = np.arange(2 * th * 2 * tw, dtype=np.float32).reshape(1, 2 * th, 2 * tw) + 1.0
        mosaic = MosaicGrid(array=volume, tile_shape=(th, tw))
        fit = MosaicFit(
            flatfields=np.ones((th, tw), np.float32),
            darkfields=np.zeros((th, tw), np.float32),
            field_mode="global",
            z_indices=[0],
        )
        out = apply_fit(mosaic, fit, n_extra_rows=2).reshape(2, th, 2, tw)
        np.testing.assert_array_equal(out[:, 0], out[:, 2])
        np.testing.assert_array_equal(out[:, 1], out[:, 2])


class TestChunkedResize:
    def test_threaded_resize_equals_inline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import linum_basic.core as core

        stack = _stack(40, 20)
        inline = BaSiC(stack)
        inline.working_size = 32
        inline.prepare()
        monkeypatch.setattr(core, "_RESIZE_PARALLEL_MIN_PIXELS", 0)
        threaded = BaSiC(stack)
        threaded.working_size = 32
        threaded.prepare()
        np.testing.assert_array_equal(threaded.img_stack_resized, inline.img_stack_resized)


class TestDarkfieldOffset:
    """The dark-field level follows the data, on every backend."""

    @staticmethod
    def _run(stack: np.ndarray, backend: str) -> BaSiC:
        model = BaSiC(stack, backend=backend, estimate_darkfield=True)
        model.working_size = stack.shape[1]
        model.run()
        return model

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    @pytest.mark.parametrize("offset", [0.0, 8.0, 20.0])
    def test_recovers_a_known_offset(self, backend: str, offset: float) -> None:
        # Images are level * flat + offset with levels from 5 upward, so the
        # stack minimum sits about 4 above the offset.  The estimate used to
        # equal that minimum whatever the true offset was.
        rng = np.random.default_rng(0)
        size = 48
        yy, xx = np.mgrid[:size, :size] / (size - 1)
        flat = 1.0 - 0.5 * ((xx - 0.45) ** 2 + (yy - 0.55) ** 2)
        level = rng.uniform(5.0, 120.0, size=(60, 1, 1))
        stack = (level * flat[None] * rng.uniform(0.97, 1.03, size=(60, size, size)) + offset).astype(np.float32)
        model = self._run(stack, backend)
        assert abs(float(model.get_darkfield().mean()) - offset) < 1.5
        assert float(stack.min()) - offset > 3.0

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_is_stable_under_a_tiny_change_of_the_input(self, backend: str) -> None:
        stack = _stack(24, 32, offset=6.0, seed=4)
        base = self._run(stack, backend)
        nudged = self._run(stack * np.float32(1.0 + 1e-6), backend)
        scale = float(stack.mean())
        assert np.abs(nudged.get_darkfield() - base.get_darkfield()).max() < 1e-3 * scale
        assert np.abs(nudged.get_flatfield() - base.get_flatfield()).max() < 1e-3


class TestDarkfieldConvergence:
    """A dark-field that is zero up to rounding does not keep the loop running."""

    @staticmethod
    def _stack_with_empty_images() -> np.ndarray:
        # Nearly rank-one images plus empty ones, as the empty tiles of a mosaic:
        # the dark-field comes out four orders of magnitude below the image
        # level, and its relative change never settles (on Torch-CPU the purely
        # relative test runs this stack to the pass cap).
        rng = np.random.default_rng(7)
        size = 32
        yy, xx = np.mgrid[:size, :size] / (size - 1)
        flat = 1.0 - 0.5 * ((xx - 0.45) ** 2 + (yy - 0.55) ** 2)
        stack = rng.uniform(20.0, 120.0, size=(40, 1, 1)) * flat[None] * rng.uniform(0.995, 1.005, size=(40, size, size))
        stack[:6] = 0.0
        return stack.astype(np.float32)

    @pytest.mark.parametrize("backend", ["numpy", "torch"])
    def test_stops_when_the_flatfield_has_converged(self, backend: str) -> None:
        stack = self._stack_with_empty_images()
        flat_only = BaSiC(stack, backend=backend)
        flat_only.working_size = 32
        flat_only.max_reweighting_iterations = 60
        flat_only.run()
        with_dark = BaSiC(stack, backend=backend, estimate_darkfield=True)
        with_dark.working_size = 32
        with_dark.max_reweighting_iterations = 60
        with_dark.run()
        assert float(np.abs(with_dark.get_darkfield()).mean()) < 1e-3 * float(stack.mean())
        assert with_dark.reweighting_iteration <= flat_only.reweighting_iteration + 2

    def test_floor_zero_restores_the_relative_test(self) -> None:
        stack = self._stack_with_empty_images()
        model = BaSiC(stack, estimate_darkfield=True)
        model.working_size = 32
        model.max_reweighting_iterations = 6
        model.prepare()
        model.darkfield_convergence_floor = 0.0
        last_flat = model.flatfield.copy()
        last_dark = np.full_like(model.darkfield, 1e-6)
        # A change as large as the (tiny) previous dark-field counts as not converged.
        model._finish_update(last_flat * (last_flat.mean() + 1e-9), last_dark * 2.0, last_flat, last_dark)
        assert model._flag_reweighting is True
        model.darkfield_convergence_floor = 0.01
        model._finish_update(last_flat * (last_flat.mean() + 1e-9), last_dark * 2.0, last_flat, last_dark)
        assert model._flag_reweighting is False


class TestMatmulPrecision:
    """Building or running a solver step leaves the process-wide matmul precision alone."""

    def test_step_builders_do_not_touch_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from linum_basic import _alm

        monkeypatch.setattr(torch, "compile", lambda fn, **kwargs: fn)
        monkeypatch.setenv("LINUM_BASIC_ALM_COMPILE_MODE", "default")
        _alm._ALM_STEP_CACHE.clear()
        _alm._ALM_STEP_BATCHED_CACHE.clear()
        before = torch.get_float32_matmul_precision()
        xp = get_xp(Backend.TORCH, "cpu")
        _alm._build_alm_step(xp, n=4, p=8, q=8, l_s=0.1)
        _alm._build_alm_step_batched(xp, z=2, n=4, p=8, q=8)
        assert torch.get_float32_matmul_precision() == before
        _alm._ALM_STEP_CACHE.clear()
        _alm._ALM_STEP_BATCHED_CACHE.clear()

    def test_a_fit_does_not_touch_it(self) -> None:
        before = torch.get_float32_matmul_precision()
        _fit(_stack(12, 16))
        assert torch.get_float32_matmul_precision() == before


class TestStepCache:
    def test_darkfield_step_is_not_reused_for_a_stack_with_another_minimum(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from linum_basic import _alm

        _legacy(monkeypatch)
        _alm._ALM_STEP_CACHE.clear()
        xp = get_xp(Backend.TORCH, "cpu")
        first = _alm._build_alm_step(xp, n=4, p=8, q=8, l_s=0.1, estimate_darkfield=True, l_d=0.1, b1_uplimit=1.0)
        second = _alm._build_alm_step(xp, n=4, p=8, q=8, l_s=0.1, estimate_darkfield=True, l_d=0.1, b1_uplimit=5.0)
        again = _alm._build_alm_step(xp, n=4, p=8, q=8, l_s=0.1, estimate_darkfield=True, l_d=0.1, b1_uplimit=1.0)
        assert first is not second
        assert first is again
        # Without dark-field the cap is unused and the step is shared.
        plain_a = _alm._build_alm_step(xp, n=4, p=8, q=8, l_s=0.1, b1_uplimit=1.0)
        plain_b = _alm._build_alm_step(xp, n=4, p=8, q=8, l_s=0.1, b1_uplimit=5.0)
        assert plain_a is plain_b
        _alm._ALM_STEP_CACHE.clear()
