import torch
from tqdm import tqdm
from typing import Any, Dict, Optional

from src.dataset.mri import fftnc, ifftnc
from src.recon_baselines.recon_baseline_resolver import BaseReconBaseline, EvaluatorFunction
from src.domain_specific_setups.natural_video_forward_models import LinearForwardModel


class LPSBaseline(BaseReconBaseline):
    """L+S reconstruction (Otazo et al. https://doi.org/10.1002/mrm.25240), using only forward_model / adjoint."""

    def __init__(
        self,
        max_iter: int = 20,
        lambda_l: float = 0.1,
        lambda_s: Optional[float] = None,
        lambda_s_by_lambda_l_factor: Optional[float] = None,
        tol: float = 1e-5,
        eval_every_iters: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        assert lambda_s is not None or lambda_s_by_lambda_l_factor is not None, \
            "Either lambda_s or lambda_s_by_lambda_l_factor must be provided"
        self.max_iter = max_iter
        self.lambda_l = lambda_l
        self.lambda_s = lambda_s
        self.lambda_s_by_lambda_l_factor = lambda_s_by_lambda_l_factor
        self.tol = tol
        self.eval_every_iters = eval_every_iters

    @torch.no_grad()
    def run(
        self,
        measurement: torch.Tensor,
        forward_model: LinearForwardModel,
        evaluator: EvaluatorFunction,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            measurement: masked k-space, shape (T, Ncoils, X, Y, 2)
            forward_model: encodes coil maps + undersampling mask, e.g. MulticoilMRIForwardModel
            evaluator: called periodically with the current (T, X, Y, 2) reconstruction estimate

        Returns:
            Reconstruction of shape (T, X, Y, 2).
        """
        lambda_s = self.lambda_s
        if lambda_s is None:
            lambda_s = self.lambda_l * self.lambda_s_by_lambda_l_factor

        d = measurement
        M = forward_model.adjoint(d)  # (T, X, Y, 2)
        S = torch.zeros_like(M)
        L = M
        res = torch.inf
        eval_results: Dict[str, Any] = {}

        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            if res <= self.tol and k > 1:
                break

            Lk = self._svt(M - S, self.lambda_l)
            Sk = self._soft_threshold_temporal(M - L, lambda_s)
            Mk = Lk + Sk - forward_model.adjoint(forward_model(Lk + Sk) - d)

            denom = (L + S).norm().clamp_min(1e-12)
            res = ((Lk + Sk) - (L + S)).norm() / denom
            M, S, L = Mk, Sk, Lk

            if evaluator is not None and k % self.eval_every_iters == 0:
                eval_results = evaluator(L + S)

            pbar.set_description(f"residual: {res.item():.3e} {eval_results}", refresh=True)

        return L + S

    def _soft_threshold_temporal(self, x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Soft-threshold the temporal Fourier transform of a (T, X, Y, 2) tensor."""
        x_c = torch.view_as_complex(x.contiguous())  # (T, X, Y)
        Tx = fftnc(x_c, dim=(0,))
        Tx_thresh = self._soft_threshold(Tx, threshold)
        return torch.view_as_real(ifftnc(Tx_thresh, dim=(0,)))

    @staticmethod
    def _soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
        return x / (x.abs() + torch.finfo(torch.float64).eps) * torch.clamp(x.abs() - threshold, min=0)

    @staticmethod
    def _svt(x: torch.Tensor, threshold: float) -> torch.Tensor:
        """Singular value thresholding of the (T, X, Y, 2) tensor's temporal low-rank structure."""
        x_c = torch.view_as_complex(x.contiguous())  # (T, X, Y)
        T, X, Y = x_c.shape
        mat = x_c.reshape(T, X * Y).permute(1, 0)  # (X*Y, T)
        U, Sv, Vh = torch.linalg.svd(mat, full_matrices=False)
        Sv_thresh = LPSBaseline._soft_threshold(Sv, Sv[0] * threshold)
        mat_thresh = U @ torch.diag(Sv_thresh.to(mat.dtype)) @ Vh
        return torch.view_as_real(mat_thresh.permute(1, 0).reshape(T, X, Y))
