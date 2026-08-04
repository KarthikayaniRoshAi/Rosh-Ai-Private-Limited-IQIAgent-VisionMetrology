from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DPDContext:
    """
    Shared context object for the complete
    Digital Part Definition (Stage-1) pipeline.
    """

    # Drawing Information
    drawing_path: Optional[Path] = None
    drawing_name: str = ""
    drawing_extension: str = ""
    drawing_size: int = 0
    drawing_revision: str = ""
    total_pages: int = 0

    # Drawing Data
    pages: List[Any] = field(default_factory=list)
    rendered_images: List[Any] = field(default_factory=list)

    # Drawing Intelligence
    raw_response: str = ""
    raw_yaml: str = ""
    parsed_yaml: Dict = field(default_factory=dict)

    # Digital Part Definition
    dpd: Dict = field(default_factory=dict)

    # Validation
    validation_status: bool = False
    validation_report: Dict = field(default_factory=dict)

    # Runtime
    metadata: Dict = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def log(self, message: str):
        self.logs.append(message)

    def warning(self, message: str):
        self.warnings.append(message)

    def error(self, message: str):
        self.errors.append(message)

    def clear(self):
        self.logs.clear()
        self.warnings.clear()
        self.errors.clear()

    def summary(self):

        print("\n" + "=" * 70)
        print("Digital Part Definition Builder")
        print("=" * 70)
        print(f"Drawing      : {self.drawing_name}")
        print(f"Pages        : {self.total_pages}")
        print(f"Validation   : {'Passed' if self.validation_status else 'Failed'}")

        if "dpd_file" in self.metadata:
            print(f"Output       : {self.metadata['dpd_file']}")

        print(f"Warnings     : {len(self.warnings)}")
        print(f"Errors       : {len(self.errors)}")
        print("=" * 70)