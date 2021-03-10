from .regularization import GaussianNoise, RescaleNormal, Invert
from .normalizers import clip, clip_and_normalize, mask_clip_and_normalize, max_min_normalize, quantile_normalize


normalizers = {
    "clip": clip,
    "clip_and_normalize": clip_and_normalize,
    "mask_clip_and_normalize": mask_clip_and_normalize,
    "max_min_normalize": max_min_normalize,
    "quantile_normalize": quantile_normalize
}


__all__ = [
    GaussianNoise,
    RescaleNormal,
    Invert,
]
