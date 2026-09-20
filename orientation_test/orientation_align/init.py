from orientation_test.orientation_align.feature_matching import FeatureMatcher
from orientation_test.orientation_align.shape_orientation import ShapeOrientationFallback
from orientation_test.orientation_align.transform import TransformEstimator
from orientation_test.orientation_align.validator import AlignmentValidator


def __init__(self):

    self.matcher = FeatureMatcher()

    self.transformer = TransformEstimator()

    self.validator = AlignmentValidator()

    # NEW FALLBACK
    self.shape_fallback = ShapeOrientationFallback(
        angle_step=1.0,
        min_score=0.70
    )