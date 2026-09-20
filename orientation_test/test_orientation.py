import cv2
import json
import sys
from pathlib import Path

import numpy as np

from orientation_align.aligner import OrientationAligner

# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "orientation_results"

ALIGNED_DIR = OUTPUT_DIR / "aligned"

OVERLAY_DIR = OUTPUT_DIR / "overlays"

MATCHING_DIR = OUTPUT_DIR / "matches"

OVERLAY_ALPHA = 0.5
MIN_CONFIDENCE = 0.30

def create_output_directories():

    ALIGNED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OVERLAY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MATCHING_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

def load_image(path: Path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise RuntimeError(
            f"Could not load image: {path}"
        )

    return image

def fetch_ground_truth_from_db(part_number: str):
    """
    Dynamically fetches the ground truth image path and JSON data 
    from your database using the provided part number.
    """
    print(f"[DB] Fetching ground truth assets for part number: {part_number}")
    
    # =========================================================================
    # TODO: Replace this comment block with your actual Django DB query.
    # Example:
    #   from myapp.models import PartGroundTruth
    #   record = PartGroundTruth.objects.get(part_number=part_number)
    #   gt_image_path = Path(record.image_path)
    #   gt_json_data = record.json_data
    # =========================================================================

    # Mock implementation fallback pointing to sample location
    gt_dir = BASE_DIR / "data" / "ground_truth"
    gt_image_path = gt_dir / "ground_truth.jpg"
    gt_json_path = gt_dir / "ground_truth.json"

    return gt_image_path, gt_json_path


def load_ground_truth_json(json_path: Path):

    if not json_path.exists():

        print(
            f"[WARNING] Ground truth JSON not found:\n"
            f"{json_path}"
        )

        return {}

    try:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception as exc:

        print(
            f"[WARNING] Could not read ground truth JSON: "
            f"{exc}"
        )

        return {}


def create_overlay(
    ground_truth,
    aligned_image
):

    # Make sure both images have the same dimensions.

    if ground_truth.shape[:2] != aligned_image.shape[:2]:

        aligned_image = cv2.resize(
            aligned_image,
            (
                ground_truth.shape[1],
                ground_truth.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    overlay = cv2.addWeighted(
        ground_truth,
        OVERLAY_ALPHA,
        aligned_image,
        1.0 - OVERLAY_ALPHA,
        0
    )

    return overlay

def create_difference(
    ground_truth,
    aligned_image
):

    if ground_truth.shape[:2] != aligned_image.shape[:2]:

        aligned_image = cv2.resize(
            aligned_image,
            (
                ground_truth.shape[1],
                ground_truth.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    gray_ref = cv2.cvtColor(
        ground_truth,
        cv2.COLOR_BGR2GRAY
    )

    gray_aligned = cv2.cvtColor(
        aligned_image,
        cv2.COLOR_BGR2GRAY
    )

    difference = cv2.absdiff(
        gray_ref,
        gray_aligned
    )

    # Normalize so differences are easier to visualize.

    difference = cv2.normalize(
        difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    return difference


def create_matching_visualization(
    reference_image,
    inspection_image,
    aligner,
    matching_data
):

    if matching_data is None:

        return None

    matches = matching_data["matches"]

    reference_keypoints = (
        matching_data["reference_keypoints"]
    )

    inspection_keypoints = (
        matching_data["inspection_keypoints"]
    )

    # Draw only the best matches.
    #
    # This prevents the output image from becoming
    # too crowded.

    matches_to_draw = matches[:100]

    visualization = cv2.drawMatches(
        reference_image,
        reference_keypoints,
        inspection_image,
        inspection_keypoints,
        matches_to_draw,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    return visualization


def process_image(
    reference_image,
    inspection_path,
    aligner,
    height
):

    inspection_image = load_image(
        inspection_path
    )

    print()
    print("=" * 70)
    print(
        f"PROCESSING: {inspection_path.name} | Height: {height}mm"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    result = aligner.align(
        reference_image,
        inspection_image
    )

    # --------------------------------------------------------
    # Basic result
    # --------------------------------------------------------

    result_data = {

        "image": inspection_path.name,
        
        "height": float(height),

        "success": bool(
            result.success
        ),

        "rotation_degrees": float(
            result.rotation_degrees
        ),

        "scale": float(
            result.scale
        ),

        "confidence": float(
            result.confidence
        ),

        "message": result.message
    }

    print(
        f"Status      : "
        f"{'SUCCESS' if result.success else 'FAILED'}"
    )

    print(
        f"Rotation    : "
        f"{result.rotation_degrees:.4f} degrees"
    )

    print(
        f"Scale       : "
        f"{result.scale:.6f}"
    )

    print(
        f"Confidence  : "
        f"{result.confidence:.4f}"
    )

    print(
        f"Message     : "
        f"{result.message}"
    )

    # --------------------------------------------------------
    # Alignment failed
    # --------------------------------------------------------

    if not result.success:

        print(
            "[WARNING] Alignment failed."
        )

        return result_data

    # --------------------------------------------------------
    # Save aligned image
    # --------------------------------------------------------

    aligned_filename = (
        inspection_path.stem +
        "_aligned.jpg"
    )

    aligned_path = (
        ALIGNED_DIR /
        aligned_filename
    )

    cv2.imwrite(
        str(aligned_path),
        result.aligned_image
    )

    result_data["aligned_image"] = str(
        aligned_path
    )

    print(
        f"Aligned image: "
        f"{aligned_path}"
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    overlay = create_overlay(
        reference_image,
        result.aligned_image
    )

    overlay_filename = (
        inspection_path.stem +
        "_overlay.jpg"
    )

    overlay_path = (
        OVERLAY_DIR /
        overlay_filename
    )

    cv2.imwrite(
        str(overlay_path),
        overlay
    )

    result_data["overlay"] = str(
        overlay_path
    )

    print(
        f"Overlay      : "
        f"{overlay_path}"
    )

    # --------------------------------------------------------
    # Create difference image
    # --------------------------------------------------------

    difference = create_difference(
        reference_image,
        result.aligned_image
    )

    difference_filename = (
        inspection_path.stem +
        "_difference.jpg"
    )

    difference_path = (
        OVERLAY_DIR /
        difference_filename
    )

    cv2.imwrite(
        str(difference_path),
        difference
    )

    result_data["difference"] = str(
        difference_path
    )

    # --------------------------------------------------------
    # Feature matching visualization
    # --------------------------------------------------------
    #
    # This is optional and only used to understand
    # whether ORB found sensible correspondences.
    #
    # We run the matcher independently here for
    # visualization.
    # --------------------------------------------------------

    try:

        matching_data = (
            aligner.matcher.match(
                reference_image,
                inspection_image
            )
        )

        matching_visualization = (
            create_matching_visualization(
                reference_image,
                inspection_image,
                aligner,
                matching_data
            )
        )

        if matching_visualization is not None:

            matching_filename = (
                inspection_path.stem +
                "_matches.jpg"
            )

            matching_path = (
                MATCHING_DIR /
                matching_filename
            )

            cv2.imwrite(
                str(matching_path),
                matching_visualization
            )

            result_data["matching_image"] = str(
                matching_path
            )

            print(
                f"Matches      : "
                f"{matching_path}"
            )

    except Exception as exc:

        print(
            f"[WARNING] Could not create "
            f"matching visualization: {exc}"
        )

    return result_data


def get_inspection_images(inspection_dir: Path):

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    }

    images = []

    for path in inspection_dir.iterdir():

        if not path.is_file():
            continue

        if path.suffix.lower() in supported_extensions:

            images.append(path)

    # Sort naturally by filename.
    images.sort(
        key=lambda p: p.name.lower()
    )

    return images


def print_summary(results):

    print()
    print()
    print("=" * 80)
    print("ORIENTATION ALIGNMENT SUMMARY")
    print("=" * 80)

    print(
        f"{'IMAGE':<18}"
        f"{'STATUS':<12}"
        f"{'ROTATION':<15}"
        f"{'SCALE':<12}"
        f"{'CONFIDENCE':<12}"
    )

    print("-" * 80)

    for result in results:

        status = (
            "PASS"
            if result["success"]
            else "FAIL"
        )

        print(
            f"{result['image']:<18}"
            f"{status:<12}"
            f"{result['rotation_degrees']:<15.3f}"
            f"{result['scale']:<12.4f}"
            f"{result['confidence']:<12.3f}"
        )

    print("-" * 80)

    successful = sum(
        1
        for result in results
        if result["success"]
    )

    failed = (
        len(results) -
        successful
    )

    print(
        f"Total images : {len(results)}"
    )

    print(
        f"Successful   : {successful}"
    )

    print(
        f"Failed       : {failed}"
    )

    print("=" * 80)


def save_report(results, ground_truth_image_path, ground_truth_json_path):

    report_path = (
        OUTPUT_DIR /
        "orientation_report.json"
    )

    report = {
        "ground_truth_image": str(
            ground_truth_image_path
        ),

        "ground_truth_json": str(
            ground_truth_json_path
        ),

        "total_images": len(results),

        "successful": sum(
            1
            for result in results
            if result["success"]
        ),

        "failed": sum(
            1
            for result in results
            if not result["success"]
        ),

        "results": results
    }

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    print()
    print(
        f"Report saved to:\n"
        f"{report_path}"
    )

def main():

    print()
    print("=" * 80)
    print("OPEN CV ORIENTATION ALIGNMENT TEST")
    print("=" * 80)

    # Simulated incoming payload properties (e.g., received from UI request)
    current_part_number = "PART-12345"
    current_height = 25.4  # mm
    inspection_dir = BASE_DIR / "inspection"

    # Step 1: Fetch ground truth image and JSON path dynamically from DB using part number
    ground_truth_image_path, ground_truth_json_path = fetch_ground_truth_from_db(current_part_number)

    if not ground_truth_image_path.exists():

        print(
            f"[ERROR] Ground truth image does not exist:\n"
            f"{ground_truth_image_path}"
        )

        sys.exit(1)

    if not inspection_dir.exists():

        print(
            f"[ERROR] Inspection directory does not exist:\n"
            f"{inspection_dir}"
        )

        sys.exit(1)

    create_output_directories()
    print()
    print(
        f"Ground truth image:\n"
        f"{ground_truth_image_path}"
    )

    reference_image = load_image(
        ground_truth_image_path
    )

    print(
        f"Ground truth resolution: "
        f"{reference_image.shape[1]} x "
        f"{reference_image.shape[0]}"
    )

    ground_truth_data = (
        load_ground_truth_json(ground_truth_json_path)
    )

    if ground_truth_data:

        print(
            "Ground truth JSON: loaded"
        )

        if isinstance(
            ground_truth_data,
            dict
        ):

            features = (
                ground_truth_data.get(
                    "features"
                )
            )

            if isinstance(
                features,
                list
            ):

                print(
                    f"Ground truth features: "
                    f"{len(features)}"
                )

    inspection_images = (
        get_inspection_images(inspection_dir)
    )

    if not inspection_images:

        print(
            "[ERROR] No inspection images found."
        )

        sys.exit(1)

    print()
    print(
        f"Inspection images found: "
        f"{len(inspection_images)}"
    )

    for image in inspection_images:

        print(
            f"  - {image.name}"
        )

    aligner = OrientationAligner()
    results = []

    for inspection_path in inspection_images:

        try:

            result = process_image(
                reference_image,
                inspection_path,
                aligner,
                current_height
            )

            results.append(result)

        except Exception as exc:

            print()
            print(
                f"[ERROR] Processing failed for "
                f"{inspection_path.name}"
            )

            print(
                f"Reason: {exc}"
            )

            results.append({

                "image": inspection_path.name,
                
                "height": float(current_height),

                "success": False,

                "rotation_degrees": 0.0,

                "scale": 1.0,

                "confidence": 0.0,

                "message": str(exc)
            })

    print_summary(
        results
    )

    save_report(
        results,
        ground_truth_image_path,
        ground_truth_json_path
    )

    print()
    print(
        "Orientation testing completed."
    )

if __name__ == "__main__":

    main()










# import cv2
# import json
# import sys
# from pathlib import Path

# import numpy as np

# from orientation_align.aligner import OrientationAligner

# # ============================================================
# # PATH CONFIGURATION
# # ============================================================

# BASE_DIR = Path(__file__).resolve().parent

# GROUND_TRUTH_DIR = BASE_DIR / "data" / "ground_truth"
# INSPECTION_DIR = BASE_DIR / "inspection"

# GROUND_TRUTH_IMAGE = (
#     GROUND_TRUTH_DIR / "ground_truth.jpg"
# )

# GROUND_TRUTH_JSON = (
#     GROUND_TRUTH_DIR / "ground_truth.json"
# )

# OUTPUT_DIR = BASE_DIR / "orientation_results"

# ALIGNED_DIR = OUTPUT_DIR / "aligned"

# OVERLAY_DIR = OUTPUT_DIR / "overlays"

# MATCHING_DIR = OUTPUT_DIR / "matches"

# OVERLAY_ALPHA = 0.5
# MIN_CONFIDENCE = 0.30

# def create_output_directories():

#     ALIGNED_DIR.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     OVERLAY_DIR.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     MATCHING_DIR.mkdir(
#         parents=True,
#         exist_ok=True
#     )

# def load_image(path: Path):

#     image = cv2.imread(
#         str(path),
#         cv2.IMREAD_COLOR
#     )

#     if image is None:

#         raise RuntimeError(
#             f"Could not load image: {path}"
#         )

#     return image

# def load_ground_truth_json():

#     if not GROUND_TRUTH_JSON.exists():

#         print(
#             f"[WARNING] Ground truth JSON not found:\n"
#             f"{GROUND_TRUTH_JSON}"
#         )

#         return {}

#     try:

#         with open(
#             GROUND_TRUTH_JSON,
#             "r",
#             encoding="utf-8"
#         ) as file:

#             return json.load(file)

#     except Exception as exc:

#         print(
#             f"[WARNING] Could not read ground truth JSON: "
#             f"{exc}"
#         )

#         return {}


# def create_overlay(
#     ground_truth,
#     aligned_image
# ):

#     # Make sure both images have the same dimensions.

#     if ground_truth.shape[:2] != aligned_image.shape[:2]:

#         aligned_image = cv2.resize(
#             aligned_image,
#             (
#                 ground_truth.shape[1],
#                 ground_truth.shape[0]
#             ),
#             interpolation=cv2.INTER_LINEAR
#         )

#     overlay = cv2.addWeighted(
#         ground_truth,
#         OVERLAY_ALPHA,
#         aligned_image,
#         1.0 - OVERLAY_ALPHA,
#         0
#     )

#     return overlay

# def create_difference(
#     ground_truth,
#     aligned_image
# ):

#     if ground_truth.shape[:2] != aligned_image.shape[:2]:

#         aligned_image = cv2.resize(
#             aligned_image,
#             (
#                 ground_truth.shape[1],
#                 ground_truth.shape[0]
#             ),
#             interpolation=cv2.INTER_LINEAR
#         )

#     gray_ref = cv2.cvtColor(
#         ground_truth,
#         cv2.COLOR_BGR2GRAY
#     )

#     gray_aligned = cv2.cvtColor(
#         aligned_image,
#         cv2.COLOR_BGR2GRAY
#     )

#     difference = cv2.absdiff(
#         gray_ref,
#         gray_aligned
#     )

#     # Normalize so differences are easier to visualize.

#     difference = cv2.normalize(
#         difference,
#         None,
#         0,
#         255,
#         cv2.NORM_MINMAX
#     )

#     return difference


# def create_matching_visualization(
#     reference_image,
#     inspection_image,
#     aligner,
#     matching_data
# ):

#     if matching_data is None:

#         return None

#     matches = matching_data["matches"]

#     reference_keypoints = (
#         matching_data["reference_keypoints"]
#     )

#     inspection_keypoints = (
#         matching_data["inspection_keypoints"]
#     )

#     # Draw only the best matches.
#     #
#     # This prevents the output image from becoming
#     # too crowded.

#     matches_to_draw = matches[:100]

#     visualization = cv2.drawMatches(
#         reference_image,
#         reference_keypoints,
#         inspection_image,
#         inspection_keypoints,
#         matches_to_draw,
#         None,
#         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
#     )

#     return visualization


# def process_image(
#     reference_image,
#     inspection_path,
#     aligner
# ):

#     inspection_image = load_image(
#         inspection_path
#     )

#     print()
#     print("=" * 70)
#     print(
#         f"PROCESSING: {inspection_path.name}"
#     )
#     print("=" * 70)

#     # --------------------------------------------------------
#     # Alignment
#     # --------------------------------------------------------

#     result = aligner.align(
#         reference_image,
#         inspection_image
#     )

#     # --------------------------------------------------------
#     # Basic result
#     # --------------------------------------------------------

#     result_data = {

#         "image": inspection_path.name,

#         "success": bool(
#             result.success
#         ),

#         "rotation_degrees": float(
#             result.rotation_degrees
#         ),

#         "scale": float(
#             result.scale
#         ),

#         "confidence": float(
#             result.confidence
#         ),

#         "message": result.message
#     }

#     print(
#         f"Status       : "
#         f"{'SUCCESS' if result.success else 'FAILED'}"
#     )

#     print(
#         f"Rotation     : "
#         f"{result.rotation_degrees:.4f} degrees"
#     )

#     print(
#         f"Scale        : "
#         f"{result.scale:.6f}"
#     )

#     print(
#         f"Confidence   : "
#         f"{result.confidence:.4f}"
#     )

#     print(
#         f"Message      : "
#         f"{result.message}"
#     )

#     # --------------------------------------------------------
#     # Alignment failed
#     # --------------------------------------------------------

#     if not result.success:

#         print(
#             "[WARNING] Alignment failed."
#         )

#         return result_data

#     # --------------------------------------------------------
#     # Save aligned image
#     # --------------------------------------------------------

#     aligned_filename = (
#         inspection_path.stem +
#         "_aligned.jpg"
#     )

#     aligned_path = (
#         ALIGNED_DIR /
#         aligned_filename
#     )

#     cv2.imwrite(
#         str(aligned_path),
#         result.aligned_image
#     )

#     result_data["aligned_image"] = str(
#         aligned_path
#     )

#     print(
#         f"Aligned image: "
#         f"{aligned_path}"
#     )

#     # --------------------------------------------------------
#     # Create overlay
#     # --------------------------------------------------------

#     overlay = create_overlay(
#         reference_image,
#         result.aligned_image
#     )

#     overlay_filename = (
#         inspection_path.stem +
#         "_overlay.jpg"
#     )

#     overlay_path = (
#         OVERLAY_DIR /
#         overlay_filename
#     )

#     cv2.imwrite(
#         str(overlay_path),
#         overlay
#     )

#     result_data["overlay"] = str(
#         overlay_path
#     )

#     print(
#         f"Overlay      : "
#         f"{overlay_path}"
#     )

#     # --------------------------------------------------------
#     # Create difference image
#     # --------------------------------------------------------

#     difference = create_difference(
#         reference_image,
#         result.aligned_image
#     )

#     difference_filename = (
#         inspection_path.stem +
#         "_difference.jpg"
#     )

#     difference_path = (
#         OVERLAY_DIR /
#         difference_filename
#     )

#     cv2.imwrite(
#         str(difference_path),
#         difference
#     )

#     result_data["difference"] = str(
#         difference_path
#     )

#     # --------------------------------------------------------
#     # Feature matching visualization
#     # --------------------------------------------------------
#     #
#     # This is optional and only used to understand
#     # whether ORB found sensible correspondences.
#     #
#     # We run the matcher independently here for
#     # visualization.
#     # --------------------------------------------------------

#     try:

#         matching_data = (
#             aligner.matcher.match(
#                 reference_image,
#                 inspection_image
#             )
#         )

#         matching_visualization = (
#             create_matching_visualization(
#                 reference_image,
#                 inspection_image,
#                 aligner,
#                 matching_data
#             )
#         )

#         if matching_visualization is not None:

#             matching_filename = (
#                 inspection_path.stem +
#                 "_matches.jpg"
#             )

#             matching_path = (
#                 MATCHING_DIR /
#                 matching_filename
#             )

#             cv2.imwrite(
#                 str(matching_path),
#                 matching_visualization
#             )

#             result_data["matching_image"] = str(
#                 matching_path
#             )

#             print(
#                 f"Matches      : "
#                 f"{matching_path}"
#             )

#     except Exception as exc:

#         print(
#             f"[WARNING] Could not create "
#             f"matching visualization: {exc}"
#         )

#     return result_data


# def get_inspection_images():

#     supported_extensions = {
#         ".jpg",
#         ".jpeg",
#         ".png",
#         ".bmp",
#         ".tif",
#         ".tiff"
#     }

#     images = []

#     for path in INSPECTION_DIR.iterdir():

#         if not path.is_file():
#             continue

#         if path.suffix.lower() in supported_extensions:

#             images.append(path)

#     # Sort naturally by filename.
#     images.sort(
#         key=lambda p: p.name.lower()
#     )

#     return images


# def print_summary(results):

#     print()
#     print()
#     print("=" * 80)
#     print("ORIENTATION ALIGNMENT SUMMARY")
#     print("=" * 80)

#     print(
#         f"{'IMAGE':<18}"
#         f"{'STATUS':<12}"
#         f"{'ROTATION':<15}"
#         f"{'SCALE':<12}"
#         f"{'CONFIDENCE':<12}"
#     )

#     print("-" * 80)

#     for result in results:

#         status = (
#             "PASS"
#             if result["success"]
#             else "FAIL"
#         )

#         print(
#             f"{result['image']:<18}"
#             f"{status:<12}"
#             f"{result['rotation_degrees']:<15.3f}"
#             f"{result['scale']:<12.4f}"
#             f"{result['confidence']:<12.3f}"
#         )

#     print("-" * 80)

#     successful = sum(
#         1
#         for result in results
#         if result["success"]
#     )

#     failed = (
#         len(results) -
#         successful
#     )

#     print(
#         f"Total images : {len(results)}"
#     )

#     print(
#         f"Successful   : {successful}"
#     )

#     print(
#         f"Failed       : {failed}"
#     )

#     print("=" * 80)


# def save_report(results):

#     report_path = (
#         OUTPUT_DIR /
#         "orientation_report.json"
#     )

#     report = {
#         "ground_truth_image": str(
#             GROUND_TRUTH_IMAGE
#         ),

#         "ground_truth_json": str(
#             GROUND_TRUTH_JSON
#         ),

#         "total_images": len(results),

#         "successful": sum(
#             1
#             for result in results
#             if result["success"]
#         ),

#         "failed": sum(
#             1
#             for result in results
#             if not result["success"]
#         ),

#         "results": results
#     }

#     with open(
#         report_path,
#         "w",
#         encoding="utf-8"
#     ) as file:

#         json.dump(
#             report,
#             file,
#             indent=4
#         )

#     print()
#     print(
#         f"Report saved to:\n"
#         f"{report_path}"
#     )

# def main():

#     print()
#     print("=" * 80)
#     print("OPEN CV ORIENTATION ALIGNMENT TEST")
#     print("=" * 80)
#     if not GROUND_TRUTH_IMAGE.exists():

#         print(
#             f"[ERROR] Ground truth image does not exist:\n"
#             f"{GROUND_TRUTH_IMAGE}"
#         )

#         sys.exit(1)

#     if not INSPECTION_DIR.exists():

#         print(
#             f"[ERROR] Inspection directory does not exist:\n"
#             f"{INSPECTION_DIR}"
#         )

#         sys.exit(1)

#     create_output_directories()
#     print()
#     print(
#         f"Ground truth image:\n"
#         f"{GROUND_TRUTH_IMAGE}"
#     )

#     reference_image = load_image(
#         GROUND_TRUTH_IMAGE
#     )

#     print(
#         f"Ground truth resolution: "
#         f"{reference_image.shape[1]} x "
#         f"{reference_image.shape[0]}"
#     )

#     ground_truth_data = (
#         load_ground_truth_json()
#     )

#     if ground_truth_data:

#         print(
#             "Ground truth JSON: loaded"
#         )

#         if isinstance(
#             ground_truth_data,
#             dict
#         ):

#             features = (
#                 ground_truth_data.get(
#                     "features"
#                 )
#             )

#             if isinstance(
#                 features,
#                 list
#             ):

#                 print(
#                     f"Ground truth features: "
#                     f"{len(features)}"
#                 )

#     inspection_images = (
#         get_inspection_images()
#     )

#     if not inspection_images:

#         print(
#             "[ERROR] No inspection images found."
#         )

#         sys.exit(1)

#     print()
#     print(
#         f"Inspection images found: "
#         f"{len(inspection_images)}"
#     )

#     for image in inspection_images:

#         print(
#             f"  - {image.name}"
#         )

#     aligner = OrientationAligner()
#     results = []

#     for inspection_path in inspection_images:

#         try:

#             result = process_image(
#                 reference_image,
#                 inspection_path,
#                 aligner
#             )

#             results.append(result)

#         except Exception as exc:

#             print()
#             print(
#                 f"[ERROR] Processing failed for "
#                 f"{inspection_path.name}"
#             )

#             print(
#                 f"Reason: {exc}"
#             )

#             results.append({

#                 "image": inspection_path.name,

#                 "success": False,

#                 "rotation_degrees": 0.0,

#                 "scale": 1.0,

#                 "confidence": 0.0,

#                 "message": str(exc)
#             })

#     print_summary(
#         results
#     )

#     save_report(
#         results
#     )

#     print()
#     print(
#         "Orientation testing completed."
#     )

# if __name__ == "__main__":

#     main()