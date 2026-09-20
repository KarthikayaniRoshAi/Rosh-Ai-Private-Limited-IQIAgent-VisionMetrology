import cv2
import numpy as np


class ShapeOrientationFallback:

    def __init__(
        self,
        angle_step=1.0,
        min_score=0.70
    ):

        self.angle_step = angle_step
        self.min_score = min_score

    # ============================================================
    # FIND GEAR
    # ============================================================

    def _find_gear(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # --------------------------------------------------------
        # Detect dark gear against bright background
        # --------------------------------------------------------

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # --------------------------------------------------------
        # Find connected components
        # --------------------------------------------------------

        num_labels, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                binary,
                connectivity=8
            )
        )

        if num_labels <= 1:
            return None

        # Ignore background.
        # Select largest dark component.
        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        gear_mask = np.uint8(
            labels == largest_label
        ) * 255

        # --------------------------------------------------------
        # Find outer contour
        # --------------------------------------------------------

        contours, _ = cv2.findContours(
            gear_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return None

        contour = max(
            contours,
            key=cv2.contourArea
        )

        # --------------------------------------------------------
        # Estimate center and radius
        # --------------------------------------------------------

        (cx, cy), radius = cv2.minEnclosingCircle(
            contour
        )

        if radius <= 10:
            return None

        return {
            "gray": gray,
            "mask": gear_mask,
            "contour": contour,
            "center": (
                float(cx),
                float(cy)
            ),
            "radius": float(radius)
        }

    # ============================================================
    # POLAR REPRESENTATION
    # ============================================================

    def _polar_image(
        self,
        gray,
        center,
        radius
    ):

        cx, cy = center

        # --------------------------------------------------------
        # Use only the internal portion of the gear.
        #
        # This intentionally reduces the influence of the
        # rotationally-symmetric outer teeth.
        # --------------------------------------------------------

        max_radius = radius * 0.90

        angular_resolution = 720

        radial_resolution = 500

        polar = cv2.warpPolar(
            gray,
            (
                radial_resolution,
                angular_resolution
            ),
            (cx, cy),
            max_radius,
            cv2.WARP_POLAR_LINEAR
        )

        # --------------------------------------------------------
        # Slight smoothing improves phase correlation.
        # --------------------------------------------------------

        polar = cv2.GaussianBlur(
            polar,
            (5, 5),
            0
        )

        return polar.astype(
            np.float32
        )

    # ============================================================
    # NORMALIZE ANGLE
    # ============================================================

    @staticmethod
    def _normalize_angle(angle):

        while angle > 180:
            angle -= 360

        while angle < -180:
            angle += 360

        return angle

    # ============================================================
    # CREATE SIMILARITY TRANSFORMATION
    # ============================================================

    def _create_matrix(
        self,
        reference_center,
        inspection_center,
        scale,
        angle
    ):

        cx_ref, cy_ref = reference_center

        cx_insp, cy_insp = inspection_center

        theta = np.radians(
            angle
        )

        cos_theta = np.cos(
            theta
        )

        sin_theta = np.sin(
            theta
        )

        # --------------------------------------------------------
        # OpenCV image-coordinate rotation convention
        #
        # x_ref = A*x_inspection + t
        # --------------------------------------------------------

        A = np.array(
            [
                [
                    scale * cos_theta,
                    scale * sin_theta
                ],
                [
                    -scale * sin_theta,
                    scale * cos_theta
                ]
            ],
            dtype=np.float32
        )

        inspection_center_vector = np.array(
            [
                cx_insp,
                cy_insp
            ],
            dtype=np.float32
        )

        reference_center_vector = np.array(
            [
                cx_ref,
                cy_ref
            ],
            dtype=np.float32
        )

        translation = (
            reference_center_vector
            -
            A @ inspection_center_vector
        )

        matrix = np.hstack(
            [
                A,
                translation.reshape(2, 1)
            ]
        )

        return matrix

    # ============================================================
    # CALCULATE ALIGNMENT SCORE
    # ============================================================

    def _calculate_score(
        self,
        reference_gray,
        aligned_image,
        reference_center,
        reference_radius
    ):

        if aligned_image is None:
            return 0.0

        if len(aligned_image.shape) == 3:

            aligned_gray = cv2.cvtColor(
                aligned_image,
                cv2.COLOR_BGR2GRAY
            )

        else:

            aligned_gray = aligned_image

        # --------------------------------------------------------
        # Compare only the internal gear area.
        #
        # This is important because the teeth themselves are
        # approximately rotationally symmetric.
        # --------------------------------------------------------

        mask = np.zeros(
            reference_gray.shape,
            dtype=np.uint8
        )

        cx, cy = reference_center

        cv2.circle(
            mask,
            (
                int(cx),
                int(cy)
            ),
            int(reference_radius * 0.90),
            255,
            -1
        )

        ref_pixels = (
            reference_gray[
                mask > 0
            ].astype(np.float32)
        )

        aligned_pixels = (
            aligned_gray[
                mask > 0
            ].astype(np.float32)
        )

        if len(ref_pixels) < 100:
            return 0.0

        # --------------------------------------------------------
        # Correlation
        # --------------------------------------------------------

        ref_std = np.std(
            ref_pixels
        )

        aligned_std = np.std(
            aligned_pixels
        )

        if ref_std < 1e-6 or aligned_std < 1e-6:
            return 0.0

        correlation = np.corrcoef(
            ref_pixels,
            aligned_pixels
        )[0, 1]

        if np.isnan(correlation):
            return 0.0

        # Convert [-1,1] → [0,1]
        score = (
            correlation + 1.0
        ) / 2.0

        return float(
            np.clip(
                score,
                0.0,
                1.0
            )
        )

    # ============================================================
    # MAIN ALIGNMENT
    # ============================================================

    def align(
        self,
        reference_image,
        inspection_image
    ):

        # --------------------------------------------------------
        # 1. Detect reference gear
        # --------------------------------------------------------

        reference = self._find_gear(
            reference_image
        )

        if reference is None:

            print(
                "[Fallback] Could not detect "
                "reference gear."
            )

            return None

        # --------------------------------------------------------
        # 2. Detect inspection gear
        # --------------------------------------------------------

        inspection = self._find_gear(
            inspection_image
        )

        if inspection is None:

            print(
                "[Fallback] Could not detect "
                "inspection gear."
            )

            return None

        reference_center = (
            reference["center"]
        )

        inspection_center = (
            inspection["center"]
        )

        reference_radius = (
            reference["radius"]
        )

        inspection_radius = (
            inspection["radius"]
        )

        # --------------------------------------------------------
        # 3. Calculate scale
        # --------------------------------------------------------

        scale = (
            reference_radius
            /
            max(
                inspection_radius,
                1e-6
            )
        )

        # --------------------------------------------------------
        # 4. Convert both images into polar coordinates
        #
        # Rotation becomes a vertical shift.
        # --------------------------------------------------------

        reference_polar = self._polar_image(
            reference["gray"],
            reference_center,
            reference_radius
        )

        inspection_polar = self._polar_image(
            inspection["gray"],
            inspection_center,
            inspection_radius
        )

        # --------------------------------------------------------
        # 5. Phase correlation
        #
        # This finds the angular shift between the two
        # internal-hole patterns.
        # --------------------------------------------------------

        shift, phase_response = (
            cv2.phaseCorrelate(
                reference_polar,
                inspection_polar
            )
        )

        angular_resolution = (
            reference_polar.shape[0]
        )

        angle = (
            shift[1]
            *
            360.0
            /
            angular_resolution
        )

        angle = self._normalize_angle(
            angle
        )

        print(
            f"Fallback phase response: "
            f"{phase_response:.4f}"
        )

        print(
            f"Fallback estimated angle: "
            f"{angle:.2f} degrees"
        )

        print(
            f"Fallback estimated scale: "
            f"{scale:.6f}"
        )

        # --------------------------------------------------------
        # 6. Create inspection -> reference matrix
        # --------------------------------------------------------

        matrix = self._create_matrix(
            reference_center,
            inspection_center,
            scale,
            angle
        )

        # --------------------------------------------------------
        # 7. Warp inspection image
        # --------------------------------------------------------

        aligned_image = cv2.warpAffine(
            inspection_image,
            matrix,
            (
                reference_image.shape[1],
                reference_image.shape[0]
            ),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255
        )

        # --------------------------------------------------------
        # 8. Calculate actual alignment score
        #
        # This checks the internal holes/slots after alignment.
        # --------------------------------------------------------

        score = self._calculate_score(
            reference["gray"],
            aligned_image,
            reference_center,
            reference_radius
        )

        print(
            f"Fallback alignment score: "
            f"{score:.4f}"
        )

        # --------------------------------------------------------
        # 9. Validate
        # --------------------------------------------------------

        if score < self.min_score:

            print(
                f"[Fallback] Alignment score too low: "
                f"{score:.4f}"
            )

            return None

        print(
            "[Fallback] Shape/polar alignment valid."
        )

        # --------------------------------------------------------
        # 10. Return result
        # --------------------------------------------------------

        return {
            "matrix": matrix,

            "aligned_image": aligned_image,

            "rotation_degrees": float(
                angle
            ),

            "scale": float(
                scale
            ),

            "confidence": float(
                score
            ),

            "message": (
                "Alignment successful "
                "using polar shape fallback"
            )
        }