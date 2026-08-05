import io
import os
import sys
import uuid
import shutil
import asyncio
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from dotenv import load_dotenv
import yaml
from pathlib import Path
load_dotenv()

from common.config_loader import ConfigLoader
from common.logger import logger
from main import print_banner

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
            print(f"⚠️ YAML file not found in candidates: {[str(p) for p in candidate_paths]}")
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

        print(f"Executing Visual Metrology engine for {len(file_paths)} drawing(s)...")

        from visual_metrology import engine
        result = engine.run(vm_config)

        #  READ & CONVERT THE OUTPUT YAML FILES TO JSON ARRAY
        parsed_yaml_results = load_generated_yaml_results(file_paths)

        EXECUTIONS_DB[execution_id]["status"] = "completed"
        EXECUTIONS_DB[execution_id]["result"] = parsed_yaml_results
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
        "result": None,
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

    return {
        "status": "completed",
        "execution_id": execution_id,
        "total_drawings": len(execution_data["result"]),
        "results": execution_data["result"]  
    }


if __name__ == "__main__":
    import uvicorn
    # Start ASGI Uvicorn server on port 8001
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=True)