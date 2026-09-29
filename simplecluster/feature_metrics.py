from dataclasses import asdict, dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureMetrics:
    original_tokens: int
    retained_tokens: int
    numerator: float
    denominator: float
    nqe_l2: float
    cosine_coverage_090: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result["cosine_coverage_at_0.90"] = result.pop("cosine_coverage_090")
        return result


def compute_feature_metrics(
    raw_features: torch.Tensor,
    compressed_features: torch.Tensor,
    *,
    chunk_size: int = 512,
    eps: float = 1e-12,
) -> FeatureMetrics:
    """Compute L2-NQE and cosine coverage without an entire N-by-B matrix.

    Both inputs contain visual *content* tokens in the same feature space,
    shaped [tokens, hidden]. Text and video newline embeddings are excluded.
    Computation uses FP32 regardless of the model's inference dtype.
    """

    if raw_features.ndim != 2 or compressed_features.ndim != 2:
        raise ValueError("Expected [tokens, hidden] feature tensors")
    if raw_features.shape[1] != compressed_features.shape[1]:
        raise ValueError("Raw and compressed feature dimensions differ")
    if raw_features.shape[0] < 1 or compressed_features.shape[0] < 1:
        raise ValueError("Both feature sets must contain at least one token")
    if chunk_size < 1 or eps <= 0:
        raise ValueError("chunk_size and eps must be positive")
    raw = raw_features.detach().float()
    compressed = compressed_features.detach().float()
    if not torch.isfinite(raw).all() or not torch.isfinite(compressed).all():
        raise ValueError("Features contain NaN or Inf")

    compressed_sq = compressed.square().sum(dim=1)
    compressed_unit = F.normalize(compressed, dim=-1, eps=1e-6)
    numerator = raw.new_zeros(())
    cosine_maxima = []
    for start in range(0, raw.shape[0], chunk_size):
        block = raw[start : start + chunk_size]
        distances = (
            block.square().sum(dim=1, keepdim=True)
            + compressed_sq.unsqueeze(0)
            - 2.0 * block @ compressed.T
        ).clamp_min_(0.0)
        numerator += distances.amin(dim=1).sum()
        cosine_maxima.append((F.normalize(block, dim=-1, eps=1e-6) @ compressed_unit.T).amax(dim=1))

    cosine_max = torch.cat(cosine_maxima)
    denominator = (raw - raw.mean(dim=0, keepdim=True)).square().sum()
    return FeatureMetrics(
        original_tokens=int(raw.shape[0]),
        retained_tokens=int(compressed.shape[0]),
        numerator=float(numerator.item()),
        denominator=float(denominator.item()),
        nqe_l2=float((numerator / (denominator + eps)).item()),
        cosine_coverage_090=float((cosine_max >= 0.90).float().mean().item()),
    )
