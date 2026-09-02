"""GP surrogate: scaled Matern 5/2 + ARD, ported from the MLS-Bench LDM agent.

Kernel, prior-mean shape and marginal-likelihood training follow the original
``ldm_agent/gp.py``. Two adaptations were forced by the target domain:

Targets are standardised
    The original scores live in ``[0, 1]``; train ``rank_ic`` lives around
    ``0.02``. With the inherited ``noise=0.05`` the observation noise would have
    exceeded the entire signal range, so history scores are z-scored before the
    fit and predictions are mapped back afterwards.

Features are standardised and the lengthscale is data-scaled
    Fingerprint coordinates span several orders of magnitude (``coverage_ratio``
    near 1, ``abs_median_log10`` near -2), so the columns are z-scored against
    the history before the kernel sees them. The inherited initial lengthscale of
    0.75 then measured as unusable: across 15 standardised dimensions the median
    pairwise distance is about 4, the Matern kernel evaluates to roughly zero
    between every pair, and the posterior collapses onto the prior mean for every
    candidate -- observed spread 3e-6, i.e. no discrimination at all. Marginal
    likelihood training did not recover from that initialisation. The lengthscale
    therefore defaults to the median pairwise distance of the standardised
    history, the usual kernel-bandwidth heuristic; on the same held-out check
    that moved correlation with the truth from -0.11 to between 0.26 and 0.48.

The prior-mean column is retained but carries no information yet: MLS-Bench fed
it a cheap proxy regret, and the label-free fingerprint has no analogue. It is
fixed at ``PRIOR_CONSTANT``, which makes ``_SigmoidPriorMean`` a constant mean.
Supplying a real cheap prior here is the obvious next improvement.
"""

from __future__ import annotations

import numpy as np
import torch
import gpytorch

_TORCH_DTYPE = torch.float64

# Neutral prior input; sigmoid(a - b * 0.5) with the trained a, b is just a
# constant mean, which is the honest default until a cheap prior signal exists.
PRIOR_CONSTANT = 0.5

_MIN_SCALE = 1e-8

# Standardised coordinates are clipped before the kernel sees them. A candidate
# whose fingerprint sits far outside the fitted cloud is genuinely unlike
# anything observed, but without a bound one bad coordinate pushes the pairwise
# distance far enough that Matern evaluates to zero for the whole batch and the
# posterior degenerates to a flat prior -- which is silent, not an error.
_STANDARDISED_CLIP = 8.0


class _SigmoidPriorMean(gpytorch.means.Mean):
    def __init__(self, prior_dim: int, a: float = 0.0, b: float = 2.0):
        super().__init__()
        self.prior_dim = int(prior_dim)
        self.a = torch.nn.Parameter(torch.tensor(a, dtype=_TORCH_DTYPE))
        self.b = torch.nn.Parameter(torch.tensor(b, dtype=_TORCH_DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x[..., self.prior_dim].clamp(0.0, 1.0)
        return torch.sigmoid(self.a - self.b * r)


class _FeatureGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_dim: int, lengthscale: float, scale: float):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = _SigmoidPriorMean(prior_dim=feature_dim)
        base = gpytorch.kernels.MaternKernel(
            nu=2.5,
            ard_num_dims=int(feature_dim),
            active_dims=tuple(range(int(feature_dim))),
        )
        base.initialize(lengthscale=float(lengthscale))
        self.covar_module = gpytorch.kernels.ScaleKernel(base)
        self.covar_module.initialize(outputscale=float(scale) ** 2)

    def forward(self, x: torch.Tensor):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


class LdmSurrogate:
    """GP surrogate over behavioural fingerprints, trained on realised scores."""

    def __init__(
        self,
        feature_dim: int,
        min_fit_data: int = 20,
        lengthscale: float | None = None,
        noise: float = 0.05,
        scale: float = 0.25,
        train_iters: int = 100,
        lr: float = 0.05,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.min_fit_data = max(1, int(min_fit_data))
        # None means "use the median heuristic against the actual history".
        self.lengthscale = float(lengthscale) if lengthscale else None
        self.noise = float(noise)
        self.scale = float(scale)
        self.train_iters = max(0, int(train_iters))
        self.lr = float(lr)
        self._model: _FeatureGP | None = None
        self._likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self._likelihood.initialize(noise=self.noise**2)
        self.trained = False
        self._feature_mean = np.zeros(self.feature_dim, dtype=float)
        self._feature_scale = np.ones(self.feature_dim, dtype=float)
        self._score_mean = 0.0
        self._score_scale = 1.0
        self.fitted_lengthscale = 1.0

    # ------------------------------------------------------------ scaling

    def _standardize_features(self, features: np.ndarray) -> np.ndarray:
        features = np.atleast_2d(np.asarray(features, dtype=float))
        standardized = (features - self._feature_mean) / self._feature_scale
        return np.clip(standardized, -_STANDARDISED_CLIP, _STANDARDISED_CLIP)

    def _build_input(self, features: np.ndarray) -> torch.Tensor:
        standardized = self._standardize_features(features)
        prior = np.full((standardized.shape[0], 1), PRIOR_CONSTANT, dtype=float)
        return torch.tensor(np.concatenate([standardized, prior], axis=1), dtype=_TORCH_DTYPE)

    # ---------------------------------------------------------------- fit

    def fit(self, features: np.ndarray, scores: np.ndarray) -> None:
        features = np.atleast_2d(np.asarray(features, dtype=float))
        scores = np.asarray(scores, dtype=float)
        if features.shape[1] != self.feature_dim:
            raise ValueError(
                f"feature dimension {features.shape[1]} does not match surrogate "
                f"dimension {self.feature_dim}; the profile schema changed"
            )

        self._feature_mean = features.mean(axis=0)
        spread = features.std(axis=0)
        # A constant coordinate carries no information; leaving its scale at 0
        # would produce NaNs, so it is neutralised instead.
        self._feature_scale = np.where(spread > _MIN_SCALE, spread, 1.0)
        self._score_mean = float(scores.mean())
        score_spread = float(scores.std())
        self._score_scale = score_spread if score_spread > _MIN_SCALE else 1.0

        self.fitted_lengthscale = (
            self.lengthscale
            if self.lengthscale is not None
            else _median_lengthscale(self._standardize_features(features))
        )

        train_x = self._build_input(features)
        train_y = torch.tensor(
            (scores - self._score_mean) / self._score_scale, dtype=_TORCH_DTYPE
        )
        model = _FeatureGP(
            train_x, train_y, self._likelihood, self.feature_dim, self.fitted_lengthscale, self.scale
        )
        if len(scores) > self.min_fit_data and self.train_iters > 0:
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._likelihood, model)
            opt = torch.optim.Adam(model.parameters(), lr=self.lr)
            model.train()
            self._likelihood.train()
            for _ in range(self.train_iters):
                opt.zero_grad()
                out = model(train_x)
                loss = -mll(out, train_y)
                loss.backward()
                opt.step()
            self.trained = True
        else:
            self.trained = False
        model.eval()
        self._likelihood.eval()
        self._model = model

    # ------------------------------------------------------------ predict

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Posterior mean and standard deviation, returned in score units."""
        features = np.atleast_2d(np.asarray(features, dtype=float))
        if self._model is None:
            # Nothing observed yet: the prior mean is flat, so every candidate
            # looks alike and the acquisition step degenerates to sampling.
            mean = np.full(features.shape[0], self._score_mean, dtype=float)
            return mean, np.full(features.shape[0], self.scale * self._score_scale, dtype=float)
        test_x = self._build_input(features)
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.lazily_evaluate_kernels(False):
            pred = self._likelihood(self._model(test_x))
            mean = pred.mean.detach().numpy()
            std = pred.variance.detach().clamp_min(1e-12).sqrt().numpy()
        return mean * self._score_scale + self._score_mean, std * self._score_scale


def _median_lengthscale(standardized: np.ndarray, floor: float = 1e-2) -> float:
    """Median pairwise distance of the standardised history.

    The kernel needs a bandwidth on the order of the distances it actually sees.
    Sub-sampled above 400 rows because the pairwise matrix is quadratic and the
    median is stable long before that.
    """
    rows = np.asarray(standardized, dtype=float)
    if rows.shape[0] < 2:
        return 1.0
    if rows.shape[0] > 400:
        step = int(np.ceil(rows.shape[0] / 400))
        rows = rows[::step]
    diff = rows[:, None, :] - rows[None, :, :]
    distances = np.sqrt((diff * diff).sum(axis=-1))
    upper = distances[np.triu_indices_from(distances, k=1)]
    upper = upper[np.isfinite(upper) & (upper > 0)]
    if upper.size == 0:
        return 1.0
    return float(max(np.median(upper), floor))
