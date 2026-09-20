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
from orientation_test.orientation_align.aligner import OrientationAligner
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
INSPECTIONS_DB: Dict[str, Dict] = {}
BASE_DIR = Path(__file__).resolve().parent

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


@app.post("/api/v1/metrology/save-plc")
async def save_position_layout(
    project_id: str = Form(...),
    views: Optional[str] = Form(None),
    view_name: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None),
    annotations: str = Form(...)  # Passed as JSON string from frontend
):

    logger.info("========== SAVE POSITION CAPTURE (FASTAPI) ==========")
    
    # Normalize view parameter name
    target_view_name = views or view_name
    uploaded_image = image or file

    # --- 1. VALIDATION ---
    if not project_id:
        raise HTTPException(status_code=400, detail="project_id is required")
    if not target_view_name:
        raise HTTPException(status_code=400, detail="views is required")
    if not uploaded_image:
        raise HTTPException(status_code=400, detail="image is required")
    if not annotations:
        raise HTTPException(status_code=400, detail="annotations is required")

    try:
        # --- 2. PARSE ANNOTATIONS JSON ---
        if isinstance(annotations, str):
            annotations_data = json.loads(annotations)
        else:
            annotations_data = annotations

        part_number = annotations_data.get('part_number', 'Unknown Part')
        drawing_id = annotations_data.get('drawingId', '')

        logger.info(f"project_id={project_id}")
        logger.info(f"view_name={target_view_name}")
        logger.info(f"part_number={part_number}")
        logger.info(f"image={uploaded_image.filename}")

        # --- 3. SAVE FILE TO DISK ---
        base_dir = Path(__file__).resolve().parent
        output_dir = base_dir / "visual_metrology" / "data" / "output" / str(project_id) / "part_layout"
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = uploaded_image.filename
        if not filename.startswith("part_layout_"):
            filename = f"part_layout_{filename}"
            
        saved_file_path = output_dir / filename

        with open(saved_file_path, "wb+") as destination:
            shutil.copyfileobj(uploaded_image.file, destination)

        logger.info(f"Layout image saved to: {saved_file_path}")

        # --- 4. OPTIONAL: SAVE METADATA RECORD LOCALLY (JSON) ---
        record_id = str(uuid.uuid4())
        record_data = {
            "id": record_id,
            "project_id": project_id,
            "view_name": target_view_name,
            "part_number": part_number,
            "drawing_id": drawing_id,
            "image_path": str(saved_file_path),
            "annotations_data": annotations_data
        }
        
        record_file_path = output_dir / f"{record_id}_meta.json"
        with open(record_file_path, "w", encoding="utf-8") as f:
            json.dump(record_data, f, indent=4)

        # --- 5. RETURN SUCCESS RESPONSE ---
        return {
            "id": record_id,
            "status": "success",
            "message": "Position capture layout saved successfully.",
            "project_id": project_id,
            "view_name": target_view_name,
            "part_number": part_number,
            "filename": filename,
            "saved_path": str(saved_file_path),
            "annotations_data": annotations_data
        }

    except json.JSONDecodeError as json_err:
        logger.exception(f"Invalid annotations JSON: {json_err}")
        raise HTTPException(
            status_code=400, 
            detail={"error": "Invalid annotations JSON", "details": str(json_err)}
        )
    except Exception as e:
        logger.exception(f"Failed to save position capture: {e}")
        raise HTTPException(
            status_code=500, 
            detail={"error": "Failed to save position capture", "details": str(e)}
        )

@app.get("/api/v1/metrology/captures")
async def list_position_captures():
    """
    Scans the visual_metrology/data/output directories and returns 
    all saved position capture records.
    """
    try:
        base_dir = Path(__file__).resolve().parent
        output_base_dir = base_dir / "visual_metrology" / "data" / "output"
        
        if not output_base_dir.exists():
            return []

        captures = []
        # Search through all project folders and meta.json files
        for meta_file in output_base_dir.glob("**/ *_meta.json"):
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    captures.append(data)
            except Exception as read_err:
                logger.warning(f"Could not read meta file {meta_file}: {read_err}")

        # Sort by creation or file system modification time (newest first)
        captures.sort(key=lambda x: str(x.get("id", "")), reverse=True)

        return {
            "status": "success",
            "count": len(captures),
            "results": captures
        }

    except Exception as e:
        logger.exception(f"Failed to list position captures: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to list position captures", "details": str(e)}
        )

@app.delete("/api/v1/metrology/captures/{record_id}")
async def delete_position_capture(record_id: str):
    """
    Deletes a specific position capture record, its metadata file, 
    and its saved layout image from disk.
    """
    try:
        base_dir = Path(__file__).resolve().parent
        output_base_dir = base_dir / "visual_metrology" / "data" / "output"
        
        if not output_base_dir.exists():
            raise HTTPException(status_code=404, detail="Capture directory not found.")

        target_meta_file = None
        target_project_dir = None

        # Locate the metadata file matching the record_id
        for meta_file in output_base_dir.glob(f"**/{record_id}_meta.json"):
            target_meta_file = meta_file
            target_project_dir = meta_file.parent
            break

        if not target_meta_file or not target_meta_file.exists():
            raise HTTPException(
                status_code=404, 
                detail=f"Position capture with ID '{record_id}' not found."
            )

        # Read meta file to locate the image path if needed
        with open(target_meta_file, "r", encoding="utf-8") as f:
            meta_data = json.load(f)
            image_path = meta_data.get("image_path")

        # Delete the image file if it exists
        if image_path and Path(image_path).exists():
            Path(image_path).unlink()
            logger.info(f"Deleted layout image: {image_path}")

        # Delete the metadata JSON file
        target_meta_file.unlink()
        logger.info(f"Deleted metadata file: {target_meta_file}")

        # Optional: Clean up project folder if it is completely empty now
        if target_project_dir and target_project_dir.exists():
            if not any(target_project_dir.iterdir()):
                shutil.rmtree(target_project_dir)
                logger.info(f"Removed empty project output directory: {target_project_dir}")

        return {
            "status": "success",
            "message": f"Position capture '{record_id}' deleted successfully."
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Failed to delete position capture {record_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to delete position capture", "details": str(e)}
        )


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


def get_part_data_from_db(part_number: str) -> Optional[Dict]:

    try:
        print(f"Fetching ground truth configuration for part: {part_number}...")
        
        # Point directly to the folder where Django saves the ground truth files
        base_ground_truth_dir = Path(r"C:\Users\karth\Downloads\main_folder\rsm\Visual Metrology\visual_partlayout\ground_truth")
        part_dir = base_ground_truth_dir / str(part_number)
        
        if not part_dir.exists():
            print(f"Ground truth directory for part {part_number} does not exist at {part_dir}")
            return None
        
        # Find the image file inside the part folder
        image_files = list(part_dir.glob("*.jpg")) + list(part_dir.glob("*.png")) + list(part_dir.glob("*.jpeg"))
        
        if not image_files:
            print(f"No ground truth image found inside {part_dir}")
            return None
            
        reference_image_path = image_files[0]
        
        # Automatically find any JSON file in the folder (handles metadata.json, part_number.json, etc.)
        drawing_config = {}
        json_files = list(part_dir.glob("*.json"))
        if json_files:
            json_path = json_files[0]  # Take the first JSON file found
            with open(json_path, "r", encoding="utf-8") as f:
                drawing_config = json.load(f)

        return {
            "reference_image_path": str(reference_image_path),
            "drawing_config": drawing_config
        }

    except Exception as e:
        print(f"Error fetching part data for {part_number}: {e}")
        raise
def run_orientation_alignment(
    inspection_image_path,
    ground_truth_image_path,
    ground_truth_json,
    part_number
):

    logger.info(
        "Orientation processing for part: %s",
        part_number
    )

    logger.info(
        "Inspection image: %s",
        inspection_image_path
    )

    logger.info(
        "Ground truth image: %s",
        ground_truth_image_path
    )

    logger.info(
        "Ground truth JSON received: %s",
        bool(ground_truth_json)
    )

    # ---------------------------------------------------------
    # Load images
    # ---------------------------------------------------------

    ground_truth_image = cv2.imread(
        str(ground_truth_image_path),
        cv2.IMREAD_COLOR
    )

    inspection_image = cv2.imread(
        str(inspection_image_path),
        cv2.IMREAD_COLOR
    )

    if ground_truth_image is None:

        raise RuntimeError(
            "Unable to load ground truth image"
        )

    if inspection_image is None:

        raise RuntimeError(
            "Unable to load inspection image"
        )

    # ---------------------------------------------------------
    # Orientation aligner
    # ---------------------------------------------------------

    aligner = OrientationAligner()

    result = aligner.align(
        ground_truth_image,
        inspection_image
    )

    # ---------------------------------------------------------
    # Save aligned image
    # ---------------------------------------------------------

    output_dir = (
        inspection_image_path.parent
        / "orientation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    aligned_path = (
        output_dir
        / f"{inspection_image_path.stem}_aligned.jpg"
    )

    if result.success:

        cv2.imwrite(
            str(aligned_path),
            result.aligned_image
        )

    # ---------------------------------------------------------
    # Return
    # ---------------------------------------------------------

    return {

        "orientation_status": (
            "success"
            if result.success
            else "failed"
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

        "message": result.message,

        "aligned_image_path": (
            str(aligned_path)
            if result.success
            else None
        )
    }


def save_temp_inspection_image(inspection_id: str, uploaded_image: UploadFile) -> Path:
    """Saves the incoming inspection image to a temporary working directory."""
    base_dir = Path(__file__).resolve().parent
    temp_dir = base_dir / "visual_metrology" / "data" / "output" / "temp_inspections"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    saved_path = temp_dir / f"{inspection_id}_{uploaded_image.filename}"
    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_image.file, buffer)
        
    return saved_path

def run_heavy_inspection_task(inspection_id: str, part_number: str, height: float, saved_image_path: Path):
    """Background task with live stdout redirection to UI log buffer for inspection."""
    INSPECTIONS_DB[inspection_id]["status"] = "processing"
    
    original_stdout = sys.stdout
    sys.stdout = ExecutionLogger(inspection_id, original_stdout) # Reusing your ExecutionLogger class

    try:
        print(f"Starting inspection process for part: {part_number}...")
        
        # 1. Fetch DB ground truth
        print("Fetching ground truth configuration from database...")
        part_data = get_part_data_from_db(part_number)
        if not part_data:
            raise ValueError(f"No ground truth data found for part '{part_number}'")
            
        ground_truth_json = part_data["drawing_config"]
        ground_truth_image_path = Path(part_data["reference_image_path"])

        # 2. Run Orientation Alignment
        print("Running orientation alignment...")
        orientation_result = run_orientation_alignment(
            inspection_image_path=saved_image_path,
            ground_truth_image_path=ground_truth_image_path,
            ground_truth_json=ground_truth_json,
            part_number=part_number,
        )

        if orientation_result["orientation_status"] != "success":
            raise RuntimeError("Could not determine part orientation.")

        # 3. Run Metric Computation with Dynamic Paths
        print("Running metrology metric computation...")
        product_height = height if height is not None else 0.0
        
        # Define dynamic paths
        target_metrics_path = Path(r"C:\Users\karth\Downloads\main_folder\rsm\Visual Metrology\visual_metrics\drawing_dimensions_new.json")
        aligned_image_path = Path(orientation_result["aligned_image_path"])
        
        # Ensure the part's JSON is copied fresh to the metrics path first (as we set up earlier)
        part_dir = Path(r"C:\Users\karth\Downloads\main_folder\rsm\Visual Metrology\visual_partlayout\ground_truth") / str(part_number)
        json_files = list(part_dir.glob("*.json"))
        if json_files:
            import json
            import shutil
            
            with open(json_files[0], "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            
            metrics_data = raw_data["annotations_data"] if "annotations_data" in raw_data else raw_data
            
            with open(target_metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics_data, f, indent=4)
        else:
            raise FileNotFoundError(f"No JSON configuration found in {part_dir}")

        # Call run_metrology_inspection passing the dynamic variables
        report_data = run_metrology_inspection(
            image_path=aligned_image_path,
            json_path=target_metrics_path,
            product_height_mm=product_height,
        )

        if isinstance(report_data, dict):
            report_data.pop("overlay_image_base64", None)

        # 4. Finalize
        INSPECTIONS_DB[inspection_id]["status"] = "completed"
        INSPECTIONS_DB[inspection_id]["results"] = {
            "part_number": part_number,
            "orientation": orientation_result,
            **(report_data or {})
        }
        print("✓ Inspection completed successfully.")

    except Exception as e:
        INSPECTIONS_DB[inspection_id]["status"] = "failed"
        INSPECTIONS_DB[inspection_id]["error"] = str(e)
        print(f"Inspection failed: {str(e)}")
        logger.error(f"Inspection {inspection_id} failed: {str(e)}")

    finally:
        sys.stdout = original_stdout

@app.post("/api/v1/metrology/inspect")
async def verify_inspection(
    background_tasks: BackgroundTasks,
    part_number: str = Form(...),
    height: float = Form(None),
    image: UploadFile = File(None),
    file: UploadFile = File(None)
):
    uploaded_image = image or file
    if not uploaded_image:
        raise HTTPException(status_code=400, detail="Image file is required.")

    # A. Flush/Delete old drawing_dimensions_new.json so it doesn't bleed over
    target_metrics_path = Path(r"C:\Users\karth\Downloads\main_folder\rsm\Visual Metrology\visual_metrics\drawing_dimensions_new.json")
    if target_metrics_path.exists():
        try:
            target_metrics_path.unlink()
        except Exception as e:
            print(f"Warning: Could not remove old metrics JSON: {e}")

    # B. Flush out the temp inspection folder (delete all old temp images)
    temp_dir = Path(r"C:\Users\karth\Downloads\main_folder\rsm\Visual Metrology\visual_metrology\data\output\temp_inspections")
    if temp_dir.exists():
        try:
            for temp_file in temp_dir.glob("*.*"):
                try:
                    temp_file.unlink()
                except Exception as file_err:
                    print(f"Could not delete temp file {temp_file.name}: {file_err}")
        except Exception as dir_err:
            print(f"Warning: Could not clear temp inspections directory: {dir_err}")

    inspection_id = str(uuid.uuid4())
    
    # 1. Save uploaded inspection image temporarily
    saved_image_path = save_temp_inspection_image(inspection_id, uploaded_image)

    # 2. Initialize tracking record
    INSPECTIONS_DB[inspection_id] = {
        "status": "processing",
        "part_number": part_number,
        "results": None,
        "error": None
    }

    # 3. Offload heavy CV pipeline to background thread
    background_tasks.add_task(
        run_heavy_inspection_task, 
        inspection_id, 
        part_number, 
        height, 
        saved_image_path
    )

    # 4. Return instant response so UI never hangs
    return {
        "status": "queued",
        "inspection_id": inspection_id,
        "message": "Inspection started in background. Poll results endpoint."
    }




# @app.post("/api/v1/metrology/inspect")
# async def verify_inspection(
#     part_number: str = Form(...),
#     height: float = Form(None),
#     image: UploadFile = File(None),
#     file: UploadFile = File(None)
# ):

#     logger.info(
#         "Inspection requested: part_number=%s",
#         part_number
#     )

#     # =========================================================
#     # 1. RECEIVE INSPECTION IMAGE
#     # =========================================================

#     uploaded_image = image or file

#     if uploaded_image is None:
#         raise HTTPException(
#             status_code=400,
#             detail="Image file is required."
#         )

#     filename = safe_filename(
#         uploaded_image.filename
#     )

#     if not is_image(filename):
#         raise HTTPException(
#             status_code=400,
#             detail=f"Unsupported image format: {filename}"
#         )

#     # =========================================================
#     # 2. FETCH GROUND TRUTH FROM DB USING PART NUMBER
#     # =========================================================

#     try:

#         part_data = get_part_data_from_db(
#             part_number
#         )

#     except Exception as db_err:

#         logger.exception(
#             "Failed to fetch part data from DB."
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=f"Database error: {str(db_err)}"
#         )

#     if not part_data:

#         raise HTTPException(
#             status_code=404,
#             detail=(
#                 f"No ground truth data found "
#                 f"for part_number '{part_number}'"
#             )
#         )

#     # =========================================================
#     # 3. EXTRACT GROUND TRUTH
#     # =========================================================

#     ground_truth_json = (
#         part_data["drawing_config"]
#     )

#     ground_truth_image_path = (
#         Path(
#             part_data["reference_image_path"]
#         )
#     )

#     logger.info(
#         "Ground truth loaded for part: %s",
#         part_number
#     )

#     logger.info(
#         "Ground truth image: %s",
#         ground_truth_image_path
#     )

#     logger.info(
#         "Ground truth drawing data loaded"
#     )

#     # =========================================================
#     # 4. SAVE CURRENT INSPECTION IMAGE
#     # =========================================================

#     part_output_dir = (
#         BASE_DIR
#         / "visual_metrology"
#         / "data"
#         / "output"
#         / part_number
#     )

#     part_output_dir.mkdir(
#         parents=True,
#         exist_ok=True
#     )

#     saved_image_path = (
#         part_output_dir / filename
#     )

#     try:

#         with saved_image_path.open("wb") as destination:

#             shutil.copyfileobj(
#                 uploaded_image.file,
#                 destination
#             )

#         logger.info(
#             "Inspection image saved: %s",
#             saved_image_path
#         )

#     except Exception as io_err:

#         logger.exception(
#             "Failed to save inspection image."
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to save image: {str(io_err)}"
#         )

#     # =========================================================
#     # 5. ORIENTATION
#     # =========================================================

#     logger.info("")
#     logger.info("=" * 70)
#     logger.info("STARTING ORIENTATION")
#     logger.info("=" * 70)

#     try:

#         orientation_result = run_orientation_alignment(
#             inspection_image_path=saved_image_path,
#             ground_truth_image_path=ground_truth_image_path,
#             ground_truth_json=ground_truth_json,
#             part_number=part_number,
#         )

#     except Exception as orientation_err:

#         logger.exception(
#             "Orientation processing failed."
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 f"Orientation processing error: "
#                 f"{str(orientation_err)}"
#             )
#         )

#     # =========================================================
#     # 6. LOG ORIENTATION RESULT
#     # =========================================================

#     logger.info("=" * 70)
#     logger.info("ORIENTATION RESULT")
#     logger.info("=" * 70)

#     logger.info(
#         "Status     : %s",
#         orientation_result["orientation_status"]
#     )

#     logger.info(
#         "Rotation   : %.4f degrees",
#         orientation_result["rotation_degrees"]
#     )

#     logger.info(
#         "Scale      : %.6f",
#         orientation_result["scale"]
#     )

#     logger.info(
#         "Confidence : %.4f",
#         orientation_result["confidence"]
#     )

#     logger.info(
#         "Aligned    : %s",
#         orientation_result["aligned_image_path"]
#     )

#     logger.info("=" * 70)

#     # =========================================================
#     # 7. CHECK ORIENTATION
#     # =========================================================

#     if (
#         orientation_result["orientation_status"]
#         != "success"
#     ):

#         raise HTTPException(
#             status_code=422,
#             detail={
#                 "message": "Could not determine part orientation",
#                 "orientation": orientation_result
#             }
#         )

#     # =========================================================
#     # 8. GET ALIGNED IMAGE
#     # =========================================================

#     aligned_image_path = Path(
#         orientation_result[
#             "aligned_image_path"
#         ]
#     )

#     # =========================================================
#     # 9. WRITE DB DRAWING DATA FOR METRICS ENGINE
#     # =========================================================

#     target_json_path = (
#         BASE_DIR
#         / "visual_metrics"
#         / "drawing_dimensions_new.json"
#     )

#     try:

#         target_json_path.parent.mkdir(
#             parents=True,
#             exist_ok=True
#         )

#         with open(
#             target_json_path,
#             "w",
#             encoding="utf-8"
#         ) as f:

#             json.dump(
#                 ground_truth_json,
#                 f,
#                 indent=4
#             )

#         logger.info(
#             "Ground truth drawing JSON "
#             "passed to metrics engine."
#         )

#     except Exception as json_err:

#         logger.exception(
#             "Failed to write drawing configuration."
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 f"Failed to prepare metric data: "
#                 f"{str(json_err)}"
#             )
#         )

#     # =========================================================
#     # 10. METRIC COMPUTATION
#     # =========================================================

#     logger.info("")
#     logger.info("=" * 70)
#     logger.info("STARTING METRIC COMPUTATION")
#     logger.info("=" * 70)

#     product_height = (
#         height if height is not None else 0.0
#     )

#     try:

#         report_data = run_metrology_inspection(
#             image_path=aligned_image_path,
#             product_height_mm=product_height,
#         )

#     except Exception as engine_err:

#         logger.exception(
#             "Metrology computation failed."
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 f"Metric computation error: "
#                 f"{str(engine_err)}"
#             )
#         )

#     logger.info(
#         "Metric computation completed."
#     )

#     # =========================================================
#     # 11. FINAL RESPONSE
#     # =========================================================

#     response = {

#         "status": "completed",

#         "inspection_id": str(
#             uuid.uuid4()
#         ),

#         "part_number": part_number,

#         "product_height_mm": product_height,

#         "input_image_path": str(
#             saved_image_path
#         ),

#         "ground_truth_image_path": str(
#             ground_truth_image_path
#         ),

#         "orientation": orientation_result
#     }

#     if isinstance(report_data, dict):

#         report_data.pop(
#             "overlay_image_base64",
#             None
#         )

#         response.update(
#             report_data
#         )

#     return response

# BASE_DIR = Path(__file__).resolve().parent

# @app.post("/api/v1/metrology/inspect",)
# async def verify_inspection(
#     part_number: str = Form(...),
#     height: float = Form(None),  
#     image: UploadFile = File(None),
#     file: UploadFile = File(None)
# ):

#     logger.info(
#         "Live inspection requested: "
#         "part_number=%s, height=%s",
#         part_number,
#         height,
#     )

#     uploaded_image = image or file

#     if uploaded_image is None:

#         raise HTTPException(
#             status_code=400,
#             detail="Image file is required.",
#         )

#     filename = safe_filename(
#         uploaded_image.filename
#     )


#     if not is_image(filename):

#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 f"Unsupported image format: "
#                 f"'{filename}'. "
#                 "Supported formats: "
#                 "jpg, jpeg, png, bmp, webp, tif, tiff."
#             ),
#         )

#     temp_dir = (
#         BASE_DIR
#         / "visual_metrology"
#         / "data"
#         / "temp"
#     )

#     temp_dir.mkdir(
#         parents=True,
#         exist_ok=True,
#     )


#     temp_filename = (
#         f"{uuid.uuid4().hex}_"
#         f"{filename}"
#     )

#     temp_image_path = (
#         temp_dir
#         / temp_filename
#     )

#     try:

#         with temp_image_path.open(
#             "wb"
#         ) as destination:

#             shutil.copyfileobj(
#                 uploaded_image.file,
#                 destination,
#             )

#         logger.info(
#             "Saved temporary inspection frame: %s",
#             temp_image_path,
#         )

#     except Exception as io_err:

#         logger.exception(
#             "Failed to save temporary inspection image."
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 "Failed to save temporary image: "
#                 f"{str(io_err)}"
#             ),
#         )


#     product_height = (
#         height
#         if height is not None
#         else 0.0
#     )


#     try:

#         report_data = run_metrology_inspection(
#             image_path=temp_image_path,
#             product_height_mm=product_height,
#         )

#     except Exception as engine_err:

#         logger.exception(
#             "Metrology computation engine failed."
#         )

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 "Inspection processing error: "
#                 f"{str(engine_err)}"
#             ),
#         )

#     finally:

#         try:

#             if temp_image_path.exists():
#                 temp_image_path.unlink()

#                 logger.info(
#                     "Deleted temporary inspection image: %s",
#                     temp_image_path,
#                 )

#         except Exception as cleanup_error:

#             logger.warning(
#                 "Failed to delete temporary image %s: %s",
#                 temp_image_path,
#                 cleanup_error,
#             )


#     response = {
#         "status": "completed",
#         "inspection_id": str(uuid.uuid4()),
#         "part_number": part_number,
#         "product_height_mm": product_height,
#     }

#     if isinstance(report_data, dict):
#         report_data.pop("overlay_image_base64", None)
#         response.update(report_data)
 
#     return response

    # Add engine-generated report fields.
    # if isinstance(report_data, dict):
    #     response.update(report_data)

    # return response

if __name__ == "__main__":
    import uvicorn
    # Start ASGI Uvicorn server on port 8001
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)


