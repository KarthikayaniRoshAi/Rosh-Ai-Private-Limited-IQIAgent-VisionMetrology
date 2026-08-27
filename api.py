import io
import json
import os
import sys
import uuid
import shutil
import asyncio
from typing import List, Dict
from fastapi import FastAPI, Form, UploadFile, File, BackgroundTasks, HTTPException
from dotenv import load_dotenv
import yaml
from pathlib import Path
load_dotenv()
from common.config_loader import ConfigLoader
from common.logger import logger
from main import print_banner
import glob

app = FastAPI(
    title="IQI™ Visual Metrology Engine API",
    description="Asynchronous processing engine for engineering drawing extraction and analysis.",
    version="0.2"
)

# In-memory execution database (Stores log buffer, state, and results per execution)
# For production scaled deployments, replace this with Redis or a DB table.
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

    # Save Captured Layout Image & Normalized JSON
@app.post("/api/v1/metrology/save-plc")
async def save_part_layout(
    project_id: str = Form(...),
    annotations: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Saves camera capture and normalized JSON annotations into visual_metrology dataset directory.
    """
    try:
        parsed_annotations = json.loads(annotations)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON format: {str(e)}")

    output_dir = os.path.join("visual_metrology", "data", "output", str(project_id), "part_layout")
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save Image File
    image_path = os.path.join(output_dir, image.filename)
    with open(image_path, "wb") as buffer:
        buffer.write(await image.read())

    # 2. Save Normalized JSON file
    base_name = os.path.splitext(image.filename)[0]
    json_path = os.path.join(output_dir, f"{base_name}_annotations.json")

    payload = {
        "project_id": project_id,
        "image_file": image.filename,
        "annotations": parsed_annotations
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "status": "success",
        "message": "Part layout and ground truth JSON saved successfully.",
        "image_path": image_path,
        "json_path": json_path
    }

@app.post("/api/v1/metrology/inspect")
async def verify_inspection(
   part_number: str = Form(...)
):
   
    logger.info(
        f"Inspection requested for part_number={part_number}"
    )

    dummy_results = [
    {
        "feature_id": "F-OD-001",
        "golden_feature_id": "F-OD",
        "feature_name": "Outer diameter",
        "measurement_type": "diameter",

        "reference": {
            "nominal": 260.0,
            "lower_limit": 259.7,
            "upper_limit": 260.1,
            "unit": "mm"
        },

        "actual": {
            "value": 259.82,
            "unit": "mm"
        },

        "deviation": -0.18,

        "coordinates": [
            {
                "left": 0.271677359063449,
                "top": 0.21783582737876625,
                "width": 0.4989741085531881,
                "height": 0.5164741212086953
            }
        ],

        "status": "PASS"
    },

    {
        "feature_id": "F-ID-001",
        "golden_feature_id": "F-ID",
        "feature_name": "Through bore",
        "measurement_type": "diameter",

        "reference": {
            "nominal": 212.0,
            "lower_limit": 211.7,
            "upper_limit": 212.1,
            "unit": "mm"
        },

        "actual": {
            "value": 212.04,
            "unit": "mm"
        },

        "deviation": 0.04,

        "coordinates": [
            {
                "left": 0.393794706683045,
                "top": 0.36314888521036526,
                "width": 0.3085760934473663,
                "height": 0.25386016127207056
            }
        ],

        "status": "PASS"
    },

    {
        "feature_id": "F-CB-001",
        "golden_feature_id": "F-HP6",
        "feature_name": "Counterbore/register diameter",
        "measurement_type": "diameter",

        "reference": {
            "nominal": 230.0,
            "lower_limit": 229.7,
            "upper_limit": 230.1,
            "unit": "mm"
        },

        "actual": {
            "value": 230.16,
            "unit": "mm"
        },

        "deviation": 0.16,

        "coordinates": [
            {
                "left": 0.30713078256591236,
                "top": 0.12154404206867052,
                "width": 0.06565448796752477,
                "height": 0.06127659065187911
            },
            {
                "left": 0.20733596085527473,
                "top": 0.21958658711167708,
                "width": 0.047271231336617814,
                "height": 0.06477811011770077
            },
            {
                "left": 0.7890347242475441,
                "top": 0.17231607432308463,
                "width": 0.04858432109596833,
                "height": 0.09279026584427405
            },
            {
                "left": 0.7956001730442965,
                "top": 0.6747841176684932,
                "width": 0.049897410855318736,
                "height": 0.08753798664554158
            },
            {
                "left": 0.6761090049434014,
                "top": 0.7763281821773215,
                "width": 0.04464505181791689,
                "height": 0.0630273503847899
            },
            {
                "left": 0.3242009494374688,
                "top": 0.7693251432456781,
                "width": 0.047271231336617814,
                "height": 0.05952583091896835
            },
            {
                "left": 0.22046685844877967,
                "top": 0.6695318384697607,
                "width": 0.04595814157726735,
                "height": 0.06477811011770085
            },
            {
                "left": 0.6656042868685975,
                "top": 0.09003036687627555,
                "width": 0.043331962058566265,
                "height": 0.05777507118605743
            }
        ],

        "status": "FAIL"
    },

    {
        "feature_id": "F-THK-001",
        "golden_feature_id": "F-THICK-BASE",
        "feature_name": "Main flange thickness",
        "measurement_type": "thickness",

        "reference": {
            "nominal": 18.0,
            "lower_limit": 17.5,
            "upper_limit": 18.5,
            "unit": "mm"
        },

        "actual": {
            "value": 18.12,
            "unit": "mm"
        },

        "deviation": 0.12,

        "coordinates": [
            {
                "left": 0.05370445901126682,
                "top": 0.00774465942946647,
                "width": 0.12343043737894653,
                "height": 0.5900060299909502
            }
        ],

        "status": "PASS"
    },

    {
        "feature_id": "F-CH-001",
        "golden_feature_id": "F-EDGE-CHAMFER-15",
        "feature_name": "Chamfer set A",
        "measurement_type": "chamfer",

        "reference": {
            "size": 1.5,
            "angle": 45.0,
            "unit": "mm / deg"
        },

        "actual": {
            "size": 1.32,
            "angle": 44.2
        },

        "deviation": {
            "size": -0.18,
            "angle": -0.8
        },

        "coordinates": [],

        "status": "FAIL"
    }
]


    passed = sum(
        1 for result in dummy_results
        if result["status"] == "PASS"
    )

    failed = sum(
        1 for result in dummy_results
        if result["status"] == "FAIL"
    )

    overall_result = "PASS" if failed == 0 else "FAIL"

    return {
        "status": "completed",

        "inspection_id": str(uuid.uuid4()),

        "part_number": part_number,

        "overall_result": overall_result,

        "summary": {
            "total_features": len(dummy_results),
            "passed": passed,
            "failed": failed
        },

        "results": dummy_results
    }

if __name__ == "__main__":
    import uvicorn
    # Start ASGI Uvicorn server on port 8001
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)



# @app.post("/api/v1/metrology/inspect")
# async def verify_inspection(
#     project_id: str = Form(...),
#     scale_mm_per_pixel: float = Form(default=0.05), # Scale: e.g., 1 pixel = 0.05 mm
#     camera_width: int = Form(default=1920),         # Image frame pixel width
#     camera_height: int = Form(default=1080)         # Image frame pixel height
# ):
#     """
#     Converts normalized fractional coordinates to mm and cross-inspects 
#     them against engineering drawing specifications and tolerances.
#     """
#     base_dir = Path(__file__).resolve().parent

#     # 1. Read Ground Truth JSON (Saved from /save-plc containing normalized fractions)
#     layout_dir = base_dir / "visual_metrology" / "data" / "output" / str(project_id) / "part_layout"
#     json_files = list(layout_dir.glob("*_annotations.json"))

#     if not json_files:
#         raise HTTPException(
#             status_code=404, 
#             detail=f"No layout annotations found for project '{project_id}' at {layout_dir}"
#         )

#     with open(json_files[0], "r", encoding="utf-8") as f:
#         ground_truth_data = json.load(f)

#     # 2. Read Engineering Drawing Specifications YAML (Generated during /train)
#     output_dir = base_dir / "visual_metrology" / "data" / "output"
#     yaml_files = list(output_dir.glob("**/*_drawing_analysis.yaml"))

#     if not yaml_files:
#         raise HTTPException(
#             status_code=404, 
#             detail="No engineering drawing analysis YAML files found in output directory."
#         )

#     with open(yaml_files[0], "r", encoding="utf-8") as f:
#         drawing_specs = yaml.safe_load(f) or {}

#     # Build tag-indexed lookup map from engineering drawing
#     spec_lookup = {}
#     for view in drawing_specs.get("views", []):
#         for feature in view.get("features", []):
#             tag_key = feature.get("tag_name") or feature.get("feature_type")
#             if tag_key:
#                 spec_lookup[tag_key.lower().strip()] = feature

#     # 3. Perform Conversion (Fraction -> Pixels -> mm) & Inspection
#     inspection_results = []
#     overall_status = "PASS"

#     annotations = ground_truth_data.get("annotations", [])

#     for annot in annotations:
#         tag_name = str(annot.get("tag_name", "")).lower().strip()
#         tag_id = annot.get("tag_id", "N/A")

#         # --- A. Extract Normalized Fraction (0.0 to 1.0) ---
#         norm_w = float(annot.get("width", 0.0))
#         norm_h = float(annot.get("height", 0.0))

#         # --- B. Convert Normalized Fraction -> Pixels ---
#         pixel_w = norm_w * camera_width
#         pixel_h = norm_h * camera_height

#         # --- C. Convert Pixels -> Millimeters (mm) ---
#         measured_pixel_size = max(pixel_w, pixel_h)
#         measured_mm = measured_pixel_size * scale_mm_per_pixel

#         # --- D. Match & Cross-Inspect against Engineering Drawing Specs ---
#         matching_spec = spec_lookup.get(tag_name)

#         if matching_spec:
#             nominal_mm = float(matching_spec.get("nominal_size", 0.0))
#             upper_tol = float(matching_spec.get("upper_deviation", 0.1))
#             lower_tol = float(matching_spec.get("lower_deviation", -0.1))

#             max_limit = nominal_mm + upper_tol
#             min_limit = nominal_mm + lower_tol

#             # Calculate Difference (Delta)
#             delta_mm = measured_mm - nominal_mm

#             # Tolerance Check
#             is_pass = min_limit <= measured_mm <= max_limit
#             feature_status = "PASS" if is_pass else "FAIL"

#             if not is_pass:
#                 overall_status = "FAIL"

#             inspection_results.append({
#                 "tag_id": tag_id,
#                 "feature_tag": annot.get("tag_name"),
#                 "normalized_fraction": {
#                     "norm_width": norm_w,
#                     "norm_height": norm_h
#                 },
#                 "calculated_pixels": round(measured_pixel_size, 2),
#                 "measured_mm": round(measured_mm, 3),
#                 "drawing_nominal_mm": round(nominal_mm, 3),
#                 "delta_mm": round(delta_mm, 3),
#                 "tolerance_range_mm": [round(min_limit, 3), round(max_limit, 3)],
#                 "status": feature_status
#             })
#         else:
#             inspection_results.append({
#                 "tag_id": tag_id,
#                 "feature_tag": annot.get("tag_name"),
#                 "normalized_fraction": {"norm_width": norm_w, "norm_height": norm_h},
#                 "measured_mm": round(measured_mm, 3),
#                 "drawing_nominal_mm": "N/A",
#                 "delta_mm": "N/A",
#                 "status": "UNMATCHED_TAG"
#             })

#     return {
#         "status": "completed",
#         "project_id": project_id,
#         "overall_result": overall_status,
#         "conversion_params": {
#             "scale_mm_per_pixel": scale_mm_per_pixel,
#             "camera_resolution": f"{camera_width}x{camera_height}"
#         },
#         "total_features_inspected": len(inspection_results),
#         "details": inspection_results
#     }

