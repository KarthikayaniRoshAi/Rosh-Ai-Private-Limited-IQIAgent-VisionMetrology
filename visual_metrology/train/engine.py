from time import perf_counter

from common.progress import Progress
from visual_metrology.context import DPDContext
from common.logger import logger
from .drawing_loader import DrawingLoader
from .drawing_intelligence import DrawingIntelligence, YAMLParser
from .validation import DPDValidator
from .export import DPDExporter,PartsMasterExporter

def run(config):
    progress = Progress()
    loader = DrawingLoader()
    start = perf_counter()

    drawings = loader.scan(
        config["dataset"]["input_directory"],
        config["dataset"]["supported_formats"]
    )

    logger.info(
        f"Discovering Engineering Drawings    "
        f"{perf_counter() - start:.2f} sec    "
        f"{len(drawings)} drawing(s) found"
    )
    
    if not drawings:
        progress.info(
            f"No supported drawings found in "
            f"{config['dataset']['input_directory']}"
        )
        return

    drawing_ai = DrawingIntelligence(config)

    for index, drawing in enumerate(drawings, start=1):
        drawing_start = perf_counter()
        context = DPDContext()

        logger.info(
            f"\nTraining Drawing {index}/{len(drawings)} : {drawing.stem}"
        )

        try:
            progress.start("Loading Engineering Drawing")
            context = loader.load(context, drawing)
            progress.complete()

            progress.start("Engineering Knowledge Extraction")
            context = drawing_ai.run(context)
            progress.complete("Engineering Knowledge Extracted")

            progress.start(
                "Building Digital Part Definition"
            )
            context = YAMLParser().run(context)
            progress.complete(
                "Digital Part Definition Generated"
            )

            progress.start("Verifying Digital Part Definition")
            context = DPDValidator().run(context)

            if context.validation_status:
                progress.complete("Verification Passed")
            else:
                progress.failed("Verification failed")

            progress.start("Publishing Training Artifacts")

            context = DPDExporter().run(
                context,
                config["output"]
            )

            context = PartsMasterExporter().run(
                context,
                config["output"]
            )            

            progress.complete("\nTraining Completed Successfully")

        except Exception as error:
            progress.failed(str(error))
            context.error(str(error))
            context.summary()
            raise
