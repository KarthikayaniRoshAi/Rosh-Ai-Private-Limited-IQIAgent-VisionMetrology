import cv2
import numpy as np


class FeatureMatcher:

    def __init__(
        self,
        n_features: int = 8000,
        ratio_test: float = 0.70
    ):

        self.ratio_test = ratio_test

        # SIFT is more robust to rotation and scale
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            contrastThreshold=0.02,
            edgeThreshold=10,
            sigma=1.6
        )

        # SIFT descriptors are floating point
        self.matcher = cv2.BFMatcher(
            cv2.NORM_L2,
            crossCheck=False
        )

    def detect(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Improve local contrast
        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        keypoints, descriptors = self.sift.detectAndCompute(
            gray,
            None
        )

        return keypoints, descriptors

    def match(
        self,
        reference_image,
        inspection_image
    ):

        ref_kp, ref_desc = self.detect(
            reference_image
        )

        insp_kp, insp_desc = self.detect(
            inspection_image
        )

        print(
            f"Reference keypoints : {len(ref_kp)}"
        )

        print(
            f"Inspection keypoints: {len(insp_kp)}"
        )

        if ref_desc is None or insp_desc is None:

            print(
                "[ERROR] Could not compute descriptors."
            )

            return None

        if len(ref_kp) < 10 or len(insp_kp) < 10:

            print(
                "[ERROR] Not enough keypoints."
            )

            return None

        matches = self.matcher.knnMatch(
            ref_desc,
            insp_desc,
            k=2
        )

        good_matches = []

        for pair in matches:

            if len(pair) != 2:
                continue

            m, n = pair

            if m.distance < self.ratio_test * n.distance:

                good_matches.append(m)

        print(
            f"Good matches        : {len(good_matches)}"
        )

        if len(good_matches) < 10:

            print(
                "[ERROR] Not enough good matches."
            )

            return None

        reference_points = np.float32([
            ref_kp[m.queryIdx].pt
            for m in good_matches
        ])

        inspection_points = np.float32([
            insp_kp[m.trainIdx].pt
            for m in good_matches
        ])

        return {
            "reference_points": reference_points,
            "inspection_points": inspection_points,
            "matches": good_matches,
            "reference_keypoints": ref_kp,
            "inspection_keypoints": insp_kp
        }