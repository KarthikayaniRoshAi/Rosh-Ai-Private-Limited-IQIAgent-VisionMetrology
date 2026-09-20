import cv2

from .feature_matching import FeatureMatcher
from .transform import TransformEstimator
from .validator import AlignmentValidator
from .models import AlignmentResult
from .shape_orientation import ShapeOrientationFallback


class OrientationAligner:

    def __init__(self):

        self.matcher = FeatureMatcher()

        self.transformer = TransformEstimator()

        self.validator = AlignmentValidator()

        # --------------------------------------------------
        # Shape-based fallback
        #
        # This is used only when the existing
        # feature-based alignment fails validation.
        # --------------------------------------------------

        self.shape_fallback = ShapeOrientationFallback(
            angle_step=1.0,
            min_score=0.70
        )

    def align(
        self,
        reference_image,
        inspection_image
    ):

        # --------------------------------------------------
        # 1. Detect and match features
        # --------------------------------------------------

        matching = self.matcher.match(
            reference_image,
            inspection_image
        )

        if matching is None:

            print(
                "[INFO] Feature matching failed."
            )

            print(
                "[INFO] Trying shape-based fallback..."
            )

            return self._try_shape_fallback(
                reference_image,
                inspection_image
            )

        # --------------------------------------------------
        # 2. Estimate transformation
        # --------------------------------------------------

        transformation = (
            self.transformer.estimate_similarity(
                matching["reference_points"],
                matching["inspection_points"]
            )
        )

        if transformation is None:

            print(
                "[INFO] Could not estimate feature-based "
                "transformation."
            )

            print(
                "[INFO] Trying shape-based fallback..."
            )

            return self._try_shape_fallback(
                reference_image,
                inspection_image
            )

        matrix = transformation["matrix"]

        confidence = transformation["confidence"]

        # --------------------------------------------------
        # 3. Validate feature-based alignment
        # --------------------------------------------------

        valid, message = self.validator.validate(
            confidence,
            transformation["inliers"]
        )

        # --------------------------------------------------
        # Feature alignment failed
        #
        # Try shape-based fallback.
        # --------------------------------------------------

        if not valid:

            print()

            print(
                "[INFO] Existing feature alignment "
                "did not pass validation."
            )

            print(
                f"[INFO] Reason: {message}"
            )

            print(
                "[INFO] Trying shape-based fallback..."
            )

            fallback_result = (
                self._try_shape_fallback(
                    reference_image,
                    inspection_image
                )
            )

            if fallback_result is not None:

                return fallback_result

            # --------------------------------------------------
            # Both algorithms failed
            # --------------------------------------------------

            return AlignmentResult(
                success=False,

                transform_matrix=matrix,

                aligned_image=None,

                rotation_degrees=0.0,

                scale=1.0,

                confidence=confidence,

                message=(
                    f"Feature alignment failed: "
                    f"{message}; "
                    f"shape fallback also failed"
                )
            )

        # --------------------------------------------------
        # 4. Extract rotation / scale
        # --------------------------------------------------

        params = self.transformer.extract_parameters(
            matrix
        )

        # --------------------------------------------------
        # 5. Warp inspection image
        #
        # IMPORTANT:
        #
        # inspection -> reference coordinate system
        # --------------------------------------------------

        aligned_image = cv2.warpAffine(
            inspection_image,
            matrix,
            (
                reference_image.shape[1],
                reference_image.shape[0]
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

        # --------------------------------------------------
        # 6. Return successful feature-based result
        # --------------------------------------------------

        return AlignmentResult(
            success=True,

            transform_matrix=matrix,

            aligned_image=aligned_image,

            rotation_degrees=params[
                "rotation_degrees"
            ],

            scale=params[
                "scale"
            ],

            confidence=confidence,

            message="Alignment successful"
        )

    # ============================================================
    # SHAPE FALLBACK
    # ============================================================

    def _try_shape_fallback(
        self,
        reference_image,
        inspection_image
    ):

        try:

            fallback = self.shape_fallback.align(
                reference_image,
                inspection_image
            )

        except Exception as exc:

            print(
                f"[ERROR] Shape fallback failed: "
                f"{exc}"
            )

            return None

        if fallback is None:

            print(
                "[WARNING] Shape fallback could not "
                "produce a valid alignment."
            )

            return None

        print()

        print(
            "[SUCCESS] Shape fallback succeeded."
        )

        print(
            f"Fallback rotation : "
            f"{fallback['rotation_degrees']:.4f} degrees"
        )

        print(
            f"Fallback scale    : "
            f"{fallback['scale']:.6f}"
        )

        print(
            f"Fallback confidence: "
            f"{fallback['confidence']:.4f}"
        )

        return AlignmentResult(
            success=True,

            transform_matrix=fallback[
                "matrix"
            ],

            aligned_image=fallback[
                "aligned_image"
            ],

            rotation_degrees=fallback[
                "rotation_degrees"
            ],

            scale=fallback[
                "scale"
            ],

            confidence=fallback[
                "confidence"
            ],

            message=fallback[
                "message"
            ]
        )