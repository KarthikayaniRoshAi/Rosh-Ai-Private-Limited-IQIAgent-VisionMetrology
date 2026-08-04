# visual_metrology/engine.py
import os
from common.logger import logger
from .train import engine as train_engine  # Pointing to your train module


def train(config):
    """Executes training with dynamic or fallback PDF list."""
    logger.info("Executing Visual Metrology Engine...")

    # 1. Use dynamically passed uploaded files (from api.py / Django)
    pdf_files = config.get("training", {}).get("active_pdf_files", [])

    # 2. Fall back to scanning folder (for local CLI / main.py testing)
    if not pdf_files:
        pdf_dir = config.get("data", {}).get(
            "training_pdf_dir", "visual_metrology/data/training"
        )
        if os.path.exists(pdf_dir):
            pdf_files = [
                os.path.join(pdf_dir, f)
                for f in os.listdir(pdf_dir)
                if f.endswith(".pdf")
            ]

    logger.info(f"Processing {len(pdf_files)} drawing file(s)...")

    # Pass updated config or file paths into your train engine
    config["training"]["active_pdf_files"] = pdf_files
    return train_engine.run(config)


def run(config):
    return train(config)




# from .train import engine as train_engine

# def train(config):
#     train_engine.run(config)

# def run(config):
#     train(config)