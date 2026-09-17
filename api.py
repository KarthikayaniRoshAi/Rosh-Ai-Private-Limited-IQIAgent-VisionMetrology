import io
import json
import os
import sys
import uuid
import shutil
import asyncio
from typing import List, Dict, Optional
from fastapi import FastAPI, Form, UploadFile, File, BackgroundTasks, HTTPException
from dotenv import load_dotenv
from fastapi.responses import FileResponse
import yaml
from pathlib import Path

from visual_metrics.metrics_computation import run_metrology_inspection
load_dotenv()
from common.config_loader import ConfigLoader
from common.logger import logger
from main import print_banner
import glob
import cv2
import numpy as np
import json
from pathlib import Path

app = FastAPI(
    title="IQI™ Visual Metrology Engine API",
    description="Asynchronous processing engine for engineering drawing extraction and analysis.",
    version="0.2"
)

EXECUTIONS_DB: Dict[str, Dict] = {}

# Load main framework config at app startup
main_config = ConfigLoader("configs/main_config.yaml").get()

def load_generated_yaml_results(file_paths: List[str]) -> List[Dict]:
    """
    Finds and converts <part_name>_drawing_analysis.yaml into an array of JSON objects.
    Uses absolute pathing to prevent working directory issues.
    """
    analysis_results = []
    
    # Get root directory of the project
    base_dir = Path(__file__).resolve().parent

    for index, file_path in enumerate(file_paths, start=1):
        # Extract base filename without extension
        base_filename = os.path.splitext(os.path.basename(file_path))[0]
        
        # Build candidate paths (checks both relative to root and relative to visual_metrology)
        candidate_paths = [
            base_dir / "visual_metrology" / "data" / "output" / base_filename / f"{base_filename}_drawing_analysis.yaml",
            base_dir / "data" / "output" / base_filename / f"{base_filename}_drawing_analysis.yaml"
        ]

        target_yaml_path = None
        for p in candidate_paths:
            if p.exists():
                target_yaml_path = p
                break

        print(f"Searching for output YAML for '{base_filename}'...")

        yaml_content = None
        if target_yaml_path and target_yaml_path.exists():
            try:
                with open(target_yaml_path, 'r', encoding='utf-8') as f:
                    yaml_content = yaml.safe_load(f)
                print(f"✓ Successfully loaded YAML: {target_yaml_path}")
            except Exception as read_err:
                print(f"Error reading YAML {target_yaml_path}: {str(read_err)}")
                yaml_content = {"error": f"Failed to parse YAML file: {str(read_err)}"}
        else:
            print(f" YAML file not found in candidates: {[str(p) for p in candidate_paths]}")
            yaml_content = {"error": f"Analysis YAML file for '{base_filename}' not found."}

        # Build JSON item object
        analysis_results.append({
            "id": f"drawing_analysis_{index}",
            "drawing_name": base_filename,
            "yaml_file_path": str(target_yaml_path) if target_yaml_path else "Not found",
            "data": yaml_content
        })

    return analysis_results

class ExecutionLogger(io.StringIO):
    def __init__(self, execution_id, original_stdout):
        super().__init__()
        self.execution_id = execution_id
        self.original_stdout = original_stdout

    def write(self, buf):
        # Keep terminal output unchanged
        self.original_stdout.write(buf)

        msg = buf.strip()
        if not msg:
            return

        if self.execution_id not in EXECUTIONS_DB:
            return

        logs = EXECUTIONS_DB[self.execution_id]["logs"]

        # Update the extraction progress instead of appending every spinner frame
        if "Engineering Knowledge Extraction..." in msg:
            if logs and logs[-1].startswith("Engineering Knowledge Extraction..."):
                logs[-1] = msg
            else:
                logs.append(msg)
        else:
            logs.append(msg)

    def flush(self):
        self.original_stdout.flush()


def process_drawings_task(execution_id: str, file_paths: List[str]):
    """Background task with live stdout redirection to UI log buffer."""
    EXECUTIONS_DB[execution_id]["status"] = "processing"
    
    # Save original stdout
    original_stdout = sys.stdout
    # Intercept print() statements from engine.run()
    sys.stdout = ExecutionLogger(execution_id, original_stdout)

    try:
        vm_config = ConfigLoader("visual_metrology/configs/config.yaml").get()
        if "training" not in vm_config:
            vm_config["training"] = {}
        vm_config["training"]["active_pdf_files"] = file_paths

        # --- CLEANUP OUTPUT DIRECTORY FOR CURRENT BATCH ---
        base_dir = Path(__file__).resolve().parent
        output_base_dir = base_dir / "visual_metrology" / "data" / "output"

        for file_path in file_paths:
            base_filename = os.path.splitext(os.path.basename(file_path))[0]
            target_out_dir = output_base_dir / base_filename
            
            # If the output directory for this part already exists from an old run, wipe it!
            if target_out_dir.exists() and target_out_dir.is_dir():
                try:
                    shutil.rmtree(target_out_dir)
                    print(f"Cleared old output directory for '{base_filename}'.")
                except Exception as clean_err:
                    print(f"Warning: Could not clear old output dir {target_out_dir}: {clean_err}")

        print(f"Executing Visual Metrology engine for {len(file_paths)} drawing(s)...")

        from visual_metrology import engine
        result = engine.run(vm_config)

        #  CONVERT THE OUTPUT YAML FILES TO JSON ARRAY
        parsed_yaml_results = load_generated_yaml_results(file_paths)

        EXECUTIONS_DB[execution_id]["status"] = "completed"
        EXECUTIONS_DB[execution_id]["results"] = parsed_yaml_results
        print("✓ Metrology extraction and analysis completed successfully.")

    except Exception as e:
        EXECUTIONS_DB[execution_id]["status"] = "failed"
        EXECUTIONS_DB[execution_id]["error"] = str(e)
        print(f"Execution failed: {str(e)}")
        logger.error(f"Execution {execution_id} failed: {str(e)}")

    finally:
        # Restore normal terminal output
        sys.stdout = original_stdout


@app.post("/api/v1/metrology/train")
async def start_metrology_training(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...)
):
    """
    HTTP POST Endpoint triggered by Django Gateway.
    Accepts PDF file uploads, registers an execution ID, and launches
    background processing immediately.
    """
    # 1. Print IQI Framework Log Banner
    print_banner(main_config, "visual_metrology", "train")
    logger.info("Received execution trigger from Django Gateway...")

    # 2. File Count Validation
    if len(files) < 1 or len(files) > 5:
        raise HTTPException(
            status_code=400, 
            detail="Provide between 1 and 5 engineering drawing PDFs."
        )

    # 3. Generate Unique Execution ID
    execution_id = str(uuid.uuid4())

    # 4. Resolve Target Directory & Save Files
    upload_dir = vm_config_dir = main_config.get("data", {}).get(
        "training_pdf_dir", "visual_metrology/data/train"
    )
    os.makedirs(upload_dir, exist_ok=True)

    # Delete old files before writing newly uploaded batch ---
    for existing_item in os.listdir(upload_dir):
        item_path = os.path.join(upload_dir, existing_item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as clean_err:
            logger.warning(f"Failed to clear old train file {item_path}: {clean_err}")

    saved_file_paths = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, 
                detail=f"File '{file.filename}' is not a valid PDF."
            )

        destination_path = os.path.join(upload_dir, file.filename)
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_file_paths.append(destination_path)
        logger.info(f"Saved incoming PDF: {file.filename} -> {destination_path}")

    # 5. Initialize Execution Tracking Record
    EXECUTIONS_DB[execution_id] = {
        "execution_id": execution_id,
        "status": "queued",
        "processed_drawings": [f.filename for f in files],
        "logs": [
            f"Execution session created (ID: {execution_id}).",
            f"Successfully uploaded {len(files)} drawing file(s) to server target."
        ],
        "results": None,
        "error": None
    }

    # 6. Dispatch Processing to Background Thread
    background_tasks.add_task(process_drawings_task, execution_id, saved_file_paths)

    # 7. Return Immediate Non-blocking Response to Gateway
    return {
        "status": "queued",
        "execution_id": execution_id,
        "message": "Metrology pipeline execution started. Poll /executions/{execution_id}/logs for status updates."
    }


@app.get("/api/v1/metrology/executions/{execution_id}/logs")
async def get_execution_logs(execution_id: str):
    """
    Polling Endpoint called by Django/UI every 3-10 seconds to retrieve 
    real-time progress logs and status for an ongoing execution session.
    """
    if execution_id not in EXECUTIONS_DB:
        raise HTTPException(
            status_code=404, 
            detail=f"Execution ID '{execution_id}' not found."
        )

    return EXECUTIONS_DB[execution_id]


@app.get("/api/v1/metrology/executions/{execution_id}/results")
async def get_execution_results(execution_id: str):
    """
    Returns the JSON array containing converted YAML drawing analysis output.
    """
    if execution_id not in EXECUTIONS_DB:
        raise HTTPException(status_code=404, detail="Execution ID not found.")

    execution_data = EXECUTIONS_DB[execution_id]

    if execution_data["status"] != "completed":
        return {
            "status": execution_data["status"],
            "message": "Results are not ready yet.",
            "results": []
        }

    results = execution_data.get("results") or []

    return {
        "status": "completed",
        "execution_id": execution_id,
        # "total_drawings": len(execution_data["result"]),
        # "results": execution_data["result"]  
        "total_drawings": len(results),
        "results": results
    }

def safe_filename(filename: Optional[str]) -> str:
    """
    Safely normalize an uploaded filename.

    Prevents paths such as:
        ../../malicious.pdf

    from escaping the intended upload directory.
    """

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    filename = Path(filename).name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid uploaded filename.",
        )

    return filename

def is_image(filename: str) -> bool:
    """Check whether the uploaded file has a supported image extension."""

    supported_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    return Path(filename).suffix.lower() in supported_extensions


def clean_output_directory(file_paths: List[str]):

    output_base_dir = (
        BASE_DIR
        / "visual_metrology"
        / "data"
        / "output"
    )

    output_base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for file_path in file_paths:

        base_filename = Path(file_path).stem

        target_out_dir = (
            output_base_dir
            / base_filename
        )

        if not target_out_dir.exists():
            continue

        if not target_out_dir.is_dir():
            continue

        try:

            shutil.rmtree(target_out_dir)

            logger.info(
                "Cleared old output directory for '%s'.",
                base_filename,
            )

        except Exception as clean_err:

            logger.warning(
                "Could not clear old output directory %s: %s",
                target_out_dir,
                clean_err,
            )


BASE_DIR = Path(__file__).resolve().parent

@app.post("/api/v1/metrology/inspect",)
async def verify_inspection(
    part_number: str = Form(...),
    height: float = Form(None),  
    image: UploadFile = File(None),
    file: UploadFile = File(None)
):

    logger.info(
        "Live inspection requested: "
        "part_number=%s, height=%s",
        part_number,
        height,
    )

    uploaded_image = image or file

    if uploaded_image is None:

        raise HTTPException(
            status_code=400,
            detail="Image file is required.",
        )

    filename = safe_filename(
        uploaded_image.filename
    )


    if not is_image(filename):

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported image format: "
                f"'{filename}'. "
                "Supported formats: "
                "jpg, jpeg, png, bmp, webp, tif, tiff."
            ),
        )

    temp_dir = (
        BASE_DIR
        / "visual_metrology"
        / "data"
        / "temp"
    )

    temp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    temp_filename = (
        f"{uuid.uuid4().hex}_"
        f"{filename}"
    )

    temp_image_path = (
        temp_dir
        / temp_filename
    )

    try:

        with temp_image_path.open(
            "wb"
        ) as destination:

            shutil.copyfileobj(
                uploaded_image.file,
                destination,
            )

        logger.info(
            "Saved temporary inspection frame: %s",
            temp_image_path,
        )

    except Exception as io_err:

        logger.exception(
            "Failed to save temporary inspection image."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save temporary image: "
                f"{str(io_err)}"
            ),
        )


    product_height = (
        height
        if height is not None
        else 0.0
    )


    try:

        report_data = run_metrology_inspection(
            image_path=temp_image_path,
            product_height_mm=product_height,
        )

    except Exception as engine_err:

        logger.exception(
            "Metrology computation engine failed."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Inspection processing error: "
                f"{str(engine_err)}"
            ),
        )

    finally:

        try:

            if temp_image_path.exists():
                temp_image_path.unlink()

                logger.info(
                    "Deleted temporary inspection image: %s",
                    temp_image_path,
                )

        except Exception as cleanup_error:

            logger.warning(
                "Failed to delete temporary image %s: %s",
                temp_image_path,
                cleanup_error,
            )


    response = {
        "status": "completed",
        "inspection_id": str(uuid.uuid4()),
        "part_number": part_number,
        "product_height_mm": product_height,
    }

    if isinstance(report_data, dict):
        report_data.pop("overlay_image_base64", None)
        response.update(report_data)
 
    return response

    # Add engine-generated report fields.
    # if isinstance(report_data, dict):
    #     response.update(report_data)

    # return response


if __name__ == "__main__":
    import uvicorn
    # Start ASGI Uvicorn server on port 8001
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)


