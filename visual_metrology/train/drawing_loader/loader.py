from pathlib import Path

from visual_metrology.context import DPDContext

class DrawingLoader:

    def scan(
        self,
        input_directory: str,
        supported_formats: list[str]
    ) -> list[Path]:

        directory = Path(input_directory)

        if not directory.exists():
            raise FileNotFoundError(directory)

        drawings = {}

        for extension in supported_formats:

            extension = extension.lower().lstrip(".")

            for drawing in directory.glob(f"*.{extension}"):

                drawings[drawing.resolve()] = drawing

        return sorted(drawings.values())

    def load(
        self,
        context: DPDContext,
        drawing: Path
    ) -> DPDContext:

        context.drawing_path = drawing
        context.drawing_name = drawing.name
        context.drawing_extension = drawing.suffix.lower()
        context.drawing_size = drawing.stat().st_size

        context.log(f"Loaded : {drawing.name}")

        return context