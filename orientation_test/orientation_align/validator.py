import numpy as np


class AlignmentValidator:

    def __init__(
        self,
        minimum_confidence=0.40,
        minimum_inliers=8
    ):

        self.minimum_confidence = minimum_confidence
        self.minimum_inliers = minimum_inliers

    def validate(
        self,
        confidence,
        inliers
    ):

        if inliers is None:

            return False, "No RANSAC inliers"

        inlier_count = int(
            np.sum(inliers)
        )

        if inlier_count < self.minimum_inliers:

            return False, (
                f"Insufficient inliers: "
                f"{inlier_count}"
            )

        if confidence < self.minimum_confidence:

            return False, (
                f"Low alignment confidence: "
                f"{confidence:.2f}"
            )

        return True, "Alignment valid"