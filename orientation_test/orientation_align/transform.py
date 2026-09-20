import cv2
import numpy as np


class TransformEstimator:

    def __init__(
        self,
        min_matches=10,
        ransac_threshold=5.0
    ):

        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold

    # ============================================================
    # ESTIMATE SIMILARITY TRANSFORMATION
    # ============================================================

    def estimate_similarity(
        self,
        reference_points,
        inspection_points
    ):

        # --------------------------------------------------------
        # Check number of matches
        # --------------------------------------------------------

        if reference_points is None or inspection_points is None:

            return None

        if len(reference_points) != len(inspection_points):

            return None

        if len(reference_points) < self.min_matches:

            return None

        # --------------------------------------------------------
        # Estimate transformation
        #
        # inspection -> reference
        #
        # estimateAffinePartial2D estimates:
        #
        #   rotation
        #   scale
        #   translation
        #
        # while restricting the transformation to similarity-like
        # motion.
        # --------------------------------------------------------

        matrix, inliers = cv2.estimateAffinePartial2D(
            inspection_points,
            reference_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            maxIters=10000,
            confidence=0.995,
            refineIters=50
        )

        if matrix is None:

            return None

        if inliers is None:

            return None

        # --------------------------------------------------------
        # Count RANSAC inliers
        # --------------------------------------------------------

        inlier_count = int(
            np.sum(inliers)
        )

        total_matches = len(
            reference_points
        )

        # --------------------------------------------------------
        # Confidence
        #
        # Confidence here is the percentage of matched points
        # that agree with the estimated transformation.
        # --------------------------------------------------------

        confidence = (
            inlier_count /
            max(total_matches, 1)
        )

        print(
            f"Total matches : {total_matches}"
        )

        print(
            f"RANSAC inliers: {inlier_count}"
        )

        print(
            f"Confidence    : {confidence:.4f}"
        )

        return {

            "matrix": matrix,

            "inliers": inliers,

            "confidence": confidence
        }

    # ============================================================
    # EXTRACT ROTATION AND SCALE
    # ============================================================

    @staticmethod
    def extract_parameters(
        matrix
    ):

        # Affine matrix:
        #
        # [ a  b  tx ]
        # [ c  d  ty ]
        #
        # For a similarity transformation:
        #
        # a ≈ s*cos(theta)
        # c ≈ s*sin(theta)

        a = matrix[0, 0]

        c = matrix[1, 0]

        # --------------------------------------------------------
        # Scale
        # --------------------------------------------------------

        scale = np.sqrt(
            a * a +
            c * c
        )

        # --------------------------------------------------------
        # Rotation
        # --------------------------------------------------------

        angle = np.degrees(
            np.arctan2(
                c,
                a
            )
        )

        return {

            "rotation_degrees": float(
                angle
            ),

            "scale": float(
                scale
            )
        }