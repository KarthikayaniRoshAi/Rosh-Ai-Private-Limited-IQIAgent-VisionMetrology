from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AlignmentResult:

    success: bool

    # 2x3 affine/similarity transformation:
    # inspection -> reference
    transform_matrix: Optional[np.ndarray]

    # Inspection image transformed into reference coordinates
    aligned_image: Optional[np.ndarray]

    # Extracted transformation parameters
    rotation_degrees: float
    scale: float

    # RANSAC inlier ratio
    confidence: float

    message: str





# from dataclasses import dataclass
# from typing import Optional
# import numpy as np


# @dataclass
# class AlignmentResult:

#     success: bool

#     # 2x3 affine/similarity transformation
#     transform_matrix: Optional[np.ndarray]

#     # Image transformed into reference coordinate system
#     aligned_image: Optional[np.ndarray]

#     # Transformation information
#     rotation_degrees: float
#     scale: float

#     # Quality of alignment
#     confidence: float

#     message: str