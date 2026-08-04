import os
import logging
from pathlib import Path

from openai import OpenAI

from common.logger import logger
from visual_metrology.context import DPDContext
from common.progress_spinner import ProgressSpinner
from .prompt import SYSTEM_PROMPT


# ----------------------------------------------------------
# Hide OpenAI / HTTP client logs from console
# ----------------------------------------------------------

logging.getLogger("httpx").disabled = True
logging.getLogger("openai").disabled = True


class DrawingIntelligence:

    def __init__(self, config=None):
        """
        Engineering Knowledge Extraction Engine
        """

        self.config = config or {}

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not configured. "
                "Please add it to the project .env file."
            )

        self.client = OpenAI(api_key=api_key)

    def run(self, context: DPDContext) -> DPDContext:

        if context.drawing_path is None:
            raise ValueError("Drawing path is empty.")

        context.log(f"Processing drawing: {context.drawing_name}")

        context.raw_response = self.process(context.drawing_path)
        context.raw_yaml = context.raw_response

        context.log("Drawing intelligence completed.")

        return context

    def process(self, drawing_path: Path) -> str:

        with open(drawing_path, "rb") as f:

            uploaded_file = self.client.files.create(
                file=f,
                purpose="assistants"
            )

        model = (
            self.config.get("gpt", {}).get("model")
            if isinstance(self.config, dict)
            else None
        ) or "gpt-5"

        spinner = ProgressSpinner(
            "Engineering Knowledge Extraction"
        )

        spinner.start()
        try:
            response = self.client.responses.create(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "input_text",
                                "text": SYSTEM_PROMPT
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_file",
                                "file_id": uploaded_file.id
                            },
                            {
                                "type": "input_text",
                                "text": (
                                    "Analyze this engineering drawing. "
                                    "Follow every instruction from the system prompt. "
                                    "Return only valid YAML."
                                )
                            }
                        ]
                    }
                ]
            )
        finally:
            spinner.stop()

        return response.output_text
