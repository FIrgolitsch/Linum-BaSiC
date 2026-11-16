r"""Device-resident inexact ALM solver for the Torch backend.

Solves the same problem as the generic step in :mod:`linum_basic._alm`, with
the same update order, but keeps every array on the solver's device for the
whole reweighting loop and touches the ``(N, P*Q)`` arrays as little as
possible.

Two things make an iteration cheap.

**Scaled dual.** The generic step accumulates the multiplier ``Y`` in float64
and divides by ``mu`` every iteration. Only ``u = Y / mu`` is ever used, and

.. math::

    u_{k+1} = \frac{\mu_k}{\mu_{k+1}} (u_k + dY_k)

so ``u`` can be carried directly. It stays at the scale of the data, which is
why float32 is enough for it.

**Implicit dual.** With ``T = D + u - B S - D_f`` and
``C = clamp(mu T, -W, W)`` the sparse residual is ``Ir = T - C / mu``, and
substituting into ``dY = D - Ir - B S_new - D_f`` gives

.. math::

    u_k + dY_k = C / \mu_k + B (S - S_{new})

The next ``u`` is therefore a scaled copy of ``C`` plus a rank-one term, so no
separate dual array is stored, and ``Ir`` and ``D - Ir`` are never formed
inside the loop: the row and column sums the updates need follow from the
sums of ``C``. An iteration without dark-field reads and writes the large
arrays seven times, against about twenty-five for the generic step.

The sorted stack, the weights and three scratch arrays are allocated once per
stack (:class:`AlmProblem`) and reused by every reweighting pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from linum_basic.backend import ArrayNamespace

__all__ = ["AlmProblem", "AlmResult", "read_alm_solver"]


def read_alm_solver() -> str:
    """Return which ALM solver the Torch backend uses.

    ``LINUM_BASIC_ALM_SOLVER=legacy`` selects the generic per-iteration step of
    :mod:`linum_basic._alm` (optionally compiled); anything else, including
    unset, selects the device-resident solver of this module.

    Returns
    -------
    str
        ``"legacy"`` or ``"device"``.
    """
    raw = os.environ.get("LINUM_BASIC_ALM_SOLVER", "device").strip().lower()
    return "legacy" if raw == "legacy" else "device"


@dataclass
class AlmResult:
    """Outcome of one ALM pass, with every field still on the device.

    Attributes
    ----------
    sf : torch.Tensor, shape (P, Q)
        Flat-field DCT coefficients.
    s_spatial : torch.Tensor, shape (1, P*Q)
        Flat-field in the spatial domain.
    b : torch.Tensor, shape (N, 1)
        Per-image baseline after the last update.
    b_ib : torch.Tensor, shape (N, 1)
        Baseline the returned ``Ib`` was built with (one update behind *b*).
    d_field_ib : torch.Tensor, shape (1, P*Q)
        Dark-field the returned ``Ib`` was built with.
    d_field : torch.Tensor, shape (1, P*Q)
        Final dark-field, with the ``B1`` offset folded in.
    iterations : int
        ALM iterations run.
    inv_mu : float
        ``1 / mu`` of the last iteration; needed to rebuild ``Ir``.
    """

    sf: Any
    s_spatial: Any
    b: Any
    b_ib: Any
    d_field_ib: Any
    d_field: Any
    iterations: int
    inv_mu: float

    def ib_mean_image(self) -> Any:
        """Return ``Ib.mean(axis=0)`` without forming ``Ib``.

        Returns
        -------
        torch.Tensor, shape (1, P*Q)
            Mean over the images of the low-rank component.
        """
        return self.s_spatial * self.b_ib.mean() + self.d_field_ib

    def ib(self) -> Any:
        """Return the full low-rank component ``S B + D_f``.

        Returns
        -------
        torch.Tensor, shape (N, P*Q)
            A new tensor on the device.
        """
        return self.s_spatial * self.b_ib + self.d_field_ib


class AlmProblem:
    """A sorted image stack on the device, ready for repeated ALM passes.

    Parameters
    ----------
    xp : ArrayNamespace
        Torch array namespace; fixes the device.
    stack : torch.Tensor, shape (N, P, Q)
        Image stack on ``xp``'s device, float32. Used as given (callers sort
        it along axis 0 beforehand when the algorithm asks for that).

    Notes
    -----
    The constants that depend only on the stack (its norm, its minimum and its
    leading singular value) are computed once here instead of once per
    reweighting pass.
    """

    def __init__(self, xp: ArrayNamespace, stack: Any) -> None:
        from linum_basic.backend import _get_dct_matrix

        torch = xp._torch
        self.xp = xp
        self._torch = torch
        n, p, q = stack.shape
        self.n, self.p, self.q = int(n), int(p), int(q)
        self.D = stack.reshape(n, p * q).contiguous()
        self.d_norm: float = xp.norm_fro(self.D)
        self.b1_uplimit: float = xp.min(self.D)
        self.sigma1: float = xp.svd_leading_singular(self.D)
        self.W = torch.ones_like(self.D)
        self._neg_w = torch.empty_like(self.D)
        self._t = torch.empty_like(self.D)
        self._c = torch.empty_like(self.D)
        self._r: Any = None  # third scratch array, allocated on first use
        device = self.D.device
        self._ap = _get_dct_matrix(self.p, device).float().contiguous()
        self._aq = _get_dct_matrix(self.q, device).float().contiguous()
        self._ap_t = self._ap.T.contiguous()
        self._aq_t = self._aq.T.contiguous()

    @classmethod
    def from_numpy(cls, xp: ArrayNamespace, stack: np.ndarray, *, sort: bool = False) -> AlmProblem:
        """Move a host stack to the device, optionally sorting it along axis 0 there.

        Parameters
        ----------
        xp : ArrayNamespace
            Torch array namespace.
        stack : numpy.ndarray, shape (N, P, Q)
            Image stack.
        sort : bool
            Sort along the stack axis on the device. This is the per-pixel
            sort BaSiC applies before fitting; on a GPU it takes milliseconds
            where ``numpy.sort`` along the first axis takes most of a second.

        Returns
        -------
        AlmProblem
            The stack on ``xp``'s device with its constants and scratch arrays.
        """
        torch = xp._torch
        with torch.inference_mode():
            dev = xp.asarray(np.ascontiguousarray(stack, dtype=np.float32))
            if sort:
                dev = torch.sort(dev, dim=0).values
        return cls(xp, dev)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _scratch(self) -> Any:
        if self._r is None:
            self._r = self._torch.empty_like(self.D)
        return self._r

    def _dct(self, x: Any) -> Any:
        return self._ap @ x @ self._aq_t

    def _idct(self, x: Any) -> Any:
        return self._ap_t @ x @ self._aq

    def _shrink(self, x: Any, threshold: Any) -> Any:
        torch = self._torch
        return torch.copysign(torch.clamp_min(x.abs() - threshold, 0.0), x)

    def set_weights(self, weight: np.ndarray | float) -> None:
        """Load the reweighting matrix.

        Parameters
        ----------
        weight : numpy.ndarray or float
            A scalar, or an array that reshapes to ``(N, P*Q)``.
        """
        torch = self._torch
        with torch.inference_mode():
            if isinstance(weight, (int, float)):
                self.W.fill_(float(weight))
            else:
                w = np.ascontiguousarray(np.asarray(weight, dtype=np.float32).reshape(self.n, self.p * self.q))
                self.W.copy_(self.xp.asarray(w))

    def sorted_stack_numpy(self) -> np.ndarray:
        """Copy the stack back to the host.

        Returns
        -------
        numpy.ndarray, shape (N, P, Q)
            The stack as held on the device.
        """
        return self.xp.to_numpy(self.D).reshape(self.n, self.p, self.q)

    # ------------------------------------------------------------------
    # One ALM pass
    # ------------------------------------------------------------------

    def solve(
        self,
        l_s: float,
        l_d: float,
        *,
        tol: float = 1e-6,
        max_iter: int = 500,
        estimate_darkfield: bool = False,
        rho: float = 1.5,
        warm_start: dict[str, Any] | None = None,
        check_every: int = 10,
        pbar: Any = None,
    ) -> AlmResult:
        """Run the inexact ALM loop with the current weights.

        Parameters
        ----------
        l_s, l_d : float
            Flat-field and dark-field regularisation weights.
        tol : float
            Stop when ``||dY||_F / ||D||_F < tol`` (tested every *check_every*
            iterations).
        max_iter : int
            Iteration cap.
        estimate_darkfield : bool
            Also update the dark-field.
        rho : float
            Growth factor of ``mu``.
        warm_start : dict or None
            Device tensors ``Sf`` ``(P, Q)``, ``B`` ``(N, 1)`` and ``D_field``
            ``(1, P*Q)`` from a previous pass. The multiplier always restarts
            at zero, as in the generic solver.
        check_every : int
            Convergence-test cadence.
        pbar : tqdm or None
            Progress bar to tick once per iteration.

        Returns
        -------
        AlmResult
            Fields of the pass, still on the device.
        """
        torch = self._torch
        n, p, q = self.n, self.p, self.q
        pq = p * q
        D, W, NW, T, C = self.D, self.W, self._neg_w, self._t, self._c
        ent2 = 10.0
        f32 = np.float32

        with torch.inference_mode():
            torch.neg(W, out=NW)
            if warm_start is not None:
                sf = warm_start["Sf"].reshape(p, q)
                b = warm_start["B"].reshape(n, 1)
                d_field = warm_start["D_field"].reshape(1, pq)
            else:
                sf = torch.zeros((p, q), dtype=D.dtype, device=D.device)
                b = torch.ones((n, 1), dtype=D.dtype, device=D.device)
                d_field = torch.zeros((1, pq), dtype=D.dtype, device=D.device)
            has_d_field = estimate_darkfield or bool(torch.any(d_field != 0))
            s = self._idct(sf).reshape(1, pq)
            b1 = torch.zeros((), dtype=D.dtype, device=D.device)

            # mu follows the generic solver in float32.
            mu = f32(12.5 / self.sigma1)
            mu_bar = f32((12.5 / self.sigma1) * 1e7)
            rho32 = f32(rho)

            # u = a_prev * C + c_prev * b_prev (x) ds_prev   (zero before the first iteration)
            a_prev = 0.0
            c_prev = 0.0
            b_prev = b
            ds_prev = s
            row_c_prev = b
            b_ib = b
            d_field_ib = d_field
            inv_mu = 0.0
            iteration = 0
            converged = False

            while not converged and iteration < max_iter:
                mu_f = float(mu)
                inv_mu = float(f32(1.0) / mu)

                # T = D + u - b (x) s - d_field
                if iteration == 0:
                    T.copy_(D)
                else:
                    torch.add(D, C, alpha=a_prev, out=T)
                    T.addcmul_(b_prev, ds_prev, value=c_prev)
                T.addcmul_(b, s, value=-1.0)
                if has_d_field:
                    T.sub_(d_field)

                # C = clamp(mu T, -W, W), so that Ir = shrink(T, W / mu) = T - C / mu.
                torch.mul(T, mu_f, out=C)
                C.clamp_(min=NW, max=W)
                col_c = C.sum(dim=0, keepdim=True)
                row_c = C.sum(dim=1, keepdim=True)

                # Flat-field: mean over images of (D - Ir - d_field + u) = (colsum(C) / mu + s sum(b)) / n.
                r_mean = (col_c * inv_mu + s * b.sum()) / n
                sf = self._shrink(self._dct(r_mean.reshape(p, q)), l_s * inv_mu)
                s_new = self._idct(sf).reshape(1, pq)

                # Baseline: row sums of D - Ir = C / mu + b (x) s + d_field - u.
                row_dmir = row_c * inv_mu + b * s.sum()
                if has_d_field:
                    row_dmir = row_dmir + d_field.sum()
                if iteration > 0:
                    row_dmir = row_dmir - (row_c_prev * a_prev + b_prev * (c_prev * ds_prev.sum()))
                r_mean_all = row_dmir.sum() / (n * pq)
                b_new = torch.clamp_min((row_dmir / pq) / (r_mean_all + 1e-9), 0.0)

                d_field_new = d_field
                dmir_ready = False
                if estimate_darkfield:
                    # The masked column sums need D - Ir itself.
                    R = self._scratch()
                    torch.sub(D, T, out=R)
                    R.add_(C, alpha=inv_mu)
                    dmir_ready = True
                    s_mean = torch.mean(s_new)
                    mask_valid_b_f = (b_new < 1.0)[:, 0].to(D.dtype)
                    mask_high_s_f = (s_new[0] > (s_mean - 1e-6)).to(D.dtype)
                    mask_low_s_f = (s_new[0] < (s_mean + 1e-6)).to(D.dtype)
                    k_cnt = torch.sum(mask_valid_b_f)
                    dminus_ir_valid_rowsum = torch.sum(R * mask_valid_b_f[:, None], dim=0, keepdim=True)
                    n_high = torch.sum(mask_high_s_f)
                    n_low = torch.sum(mask_low_s_f)
                    # One bright-minus-dim contrast per image (see the generic step).
                    r_high = torch.sum(R * mask_high_s_f[None, :], dim=1) / (n_high + 1e-9)
                    r_low = torch.sum(R * mask_low_s_f[None, :], dim=1) / (n_low + 1e-9)
                    b1_cand = (r_high - r_low) / (r_mean_all + 1e-9)
                    b_flat = b_new[:, 0]
                    sum_b2 = torch.sum(b_flat**2 * mask_valid_b_f)
                    sum_b = torch.sum(b_flat * mask_valid_b_f)
                    temp3 = torch.sum(b1_cand * mask_valid_b_f)
                    temp4 = torch.sum(b_flat * b1_cand * mask_valid_b_f)
                    denom = sum_b * temp3 - k_cnt * temp4
                    b1_new_raw = (sum_b2 * temp3 - sum_b * temp4) / (denom + 1e-30)
                    b1_new_guarded = torch.where(torch.abs(denom) > 1e-30, b1_new_raw, b1)
                    b1_cap = self.b1_uplimit / (s_mean + 1e-9)
                    b1_new = torch.minimum(b1_new_guarded, b1_cap)
                    b1 = torch.where(b1_new > 0.0, b1_new, b1)
                    z = b1 * (s_mean - s_new)
                    k_cnt_f64 = k_cnt.double()
                    b_valid_mean = sum_b.double() / (k_cnt_f64 + 1e-30)
                    a1_offset = dminus_ir_valid_rowsum.double() / (k_cnt_f64 + 1e-30) - b_valid_mean * s_new.double()
                    a_offset = a1_offset - z.double()
                    a_offset_mean = torch.mean(a_offset)
                    a_offset_centered = (a_offset - a_offset_mean).to(D.dtype)
                    dr_f = self._dct(a_offset_centered.reshape(p, q))
                    dr_f_shrunk = self._shrink(dr_f, l_d * inv_mu / ent2)
                    dr = self._idct(dr_f_shrunk).reshape(1, pq)
                    dr = self._shrink(dr, l_d * inv_mu / ent2)
                    d_field_new = (dr + a_offset_mean.to(D.dtype) + z).to(D.dtype)

                iteration += 1
                mu_next = min(mu * rho32, mu_bar)
                c = float(mu / mu_next)

                if iteration % check_every == 0:
                    # dY = D - Ir - b (x) s_new - d_field
                    R = self._scratch()
                    if not dmir_ready:
                        torch.sub(D, T, out=R)
                        R.add_(C, alpha=inv_mu)
                    R.addcmul_(b, s_new, value=-1.0)
                    if has_d_field:
                        R.sub_(d_field)
                    if float(torch.linalg.norm(R)) / (self.d_norm + 1e-9) < tol:
                        converged = True

                # Shift state: the next u is (mu / mu_next) (C / mu + b (x) (s - s_new)).
                b_ib = b
                d_field_ib = d_field
                a_prev = c * inv_mu
                c_prev = c
                b_prev = b
                ds_prev = s - s_new
                row_c_prev = row_c
                s = s_new
                b = b_new
                d_field = d_field_new
                mu = mu_next
                if pbar is not None:
                    pbar.update()

            d_field_final = (d_field + b1 * s).to(D.dtype)

        return AlmResult(
            sf=sf,
            s_spatial=s,
            b=b,
            b_ib=b_ib,
            d_field_ib=d_field_ib,
            d_field=d_field_final,
            iterations=iteration,
            inv_mu=inv_mu,
        )

    # ------------------------------------------------------------------
    # After a pass
    # ------------------------------------------------------------------

    def residual(self, result: AlmResult) -> Any:
        """Return the sparse residual ``Ir`` of the pass that produced *result*.

        Rebuilt from the scratch arrays of that pass into a third scratch
        array, so it is valid until the next call to :meth:`solve`,
        :meth:`residual` or :meth:`reweight`.

        Parameters
        ----------
        result : AlmResult
            Result of the most recent :meth:`solve`.

        Returns
        -------
        torch.Tensor, shape (N, P*Q)
            A scratch array owned by the problem; copy it to keep it.
        """
        torch = self._torch
        with torch.inference_mode():
            R = self._scratch()
            if result.iterations == 0:
                R.zero_()
            else:
                torch.add(self._t, self._c, alpha=-result.inv_mu, out=R)
        return R

    def reweight(self, result: AlmResult, ib_mean: float, epsilon: float) -> None:
        """Set the weights for the next pass: ``W = 1 / (|Ir / (mean(Ib) + 1e-6)| + epsilon)``, mean one.

        Parameters
        ----------
        result : AlmResult
            The pass whose residual drives the weights.
        ib_mean : float
            Mean of the low-rank component ``Ib`` of that pass.
        epsilon : float
            Stability constant of the reweighted-L1 update.
        """
        torch = self._torch
        with torch.inference_mode():
            R = self.residual(result)
            R.div_(ib_mean + 1e-6).abs_().add_(epsilon)
            torch.reciprocal(R, out=self.W)
            self.W.mul_(self.W.numel() / float(self.W.sum()))
