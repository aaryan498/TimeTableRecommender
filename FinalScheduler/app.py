from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List
import json
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import tempfile
from pathlib import Path

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic here
    if USE_CEREBRAS:
        print("Using Cerebras Inference")
    elif llm_extractor:
        print("Using Gemini API")
    else:
        print("WARNING: No LLM extraction available!")
        print("   Please set CEREBRAS_API_KEY or GEMINI_API_KEY")
        print("   in your environment variables.")

    if USE_CEREBRAS:
        print("\nCerebras Ultra-Fast Extraction Ready!")
    elif llm_extractor:
        print("\nLLM extraction is ready")
    else:
        print("\nLLM extraction disabled")

    print("\nAPI Documentation available at:")
    print("Swagger UI: http://localhost:8000/docs")
    print("ReDoc: http://localhost:8000/redoc")
    print("="*60 + "\n")
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Smart Timetable Generator API",
    description="Ultra-fast timetable generation with Cerebras/Gemini LLM",
    version="3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# App configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.xlsx', '.xls', '.xlsm', '.json'}

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class GenerateRequest(BaseModel):
    """Request model for timetable generation"""
    config: Optional[Dict[str, Any]] = None

class EventsData(BaseModel):
    """Model for events data"""
    events: List[Dict[str, Any]]

class DynamicUpdateRequest(BaseModel):
    """Request model for dynamic updates"""
    events: Optional[Dict[str, Any]] = None
    use_existing_timetable: bool = True

class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    status: str
    timestamp: Optional[str] = None
    error: Optional[str] = None
    has_parsed_config: bool
    has_results: bool
    llm_backend: str

# =============================================================================
# ULTRA-FAST LLM EXTRACTOR SETUP WITH CEREBRAS + GEMINI FALLBACK
# =============================================================================

# Initialize LLM-based timetable extractor with priority order
llm_extractor = None
USE_CEREBRAS = False

# Try Cerebras first (ultra-fast)
try:
    from timetable_extractor_gemini import TimetableExtractor
    CEREBRAS_API_KEY = os.getenv('CEREBRAS_API_KEY')
    
    if CEREBRAS_API_KEY:
        llm_extractor = TimetableExtractor(CEREBRAS_API_KEY)
        USE_CEREBRAS = True
        logger.info("Using Cerebras for ultra-fast extraction")
    else:
        logger.warning("Cerebras API key not available")
except ImportError:
    logger.info("Cerebras extractor not available, trying Gemini")
except Exception as e:
    logger.warning(f"Cerebras initialization failed: {e}")

# Fallback to Gemini if Cerebras not available
if not llm_extractor:
    try:
        from timetable_extractor_gemini import TimetableExtractor
        GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
        
        if GEMINI_API_KEY:
            llm_extractor = TimetableExtractor(GEMINI_API_KEY)
            logger.info("Gemini LLM extractor initialized successfully")
            print("Using Gemini LLM extraction as fallback")
        else:
            logger.warning("Gemini API key not available - LLM extraction disabled")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini extractor: {e}")
        llm_extractor = None

# Import timetable modules
try:
    from timetable_generator import GeneticAlgorithm, TimetableData, TimetableExporter
    from dynamic_updater import DynamicUpdater
except ImportError as e:
    logger.error(f"Failed to import timetable modules: {e}")

# =============================================================================
# GLOBAL STATE MANAGEMENT
# =============================================================================

# Store dynamic update results
latest_dynamic_update = {
    "status": "not_started",
    "timestamp": None,
    "data": None,
    "events": None,
    "original_timetable": None
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def allowed_file(filename: str) -> bool:
    """Check if uploaded file has allowed extension"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

async def generate_timetables_async(config_data: Optional[Dict] = None):
    """
    Generate timetables in background (DEPRECATED)
    This function is no longer used as all generation is now synchronous
    """
    logger.info("Background generation function deprecated - no longer supported")

async def apply_dynamic_updates_async(events_data: Dict, base_timetable: Dict, config_source: Dict):
    """Apply dynamic updates in background with provided timetable and config"""
    global latest_dynamic_update
    try:
        latest_dynamic_update["status"] = "in_progress"
        latest_dynamic_update["timestamp"] = datetime.now().isoformat()
        latest_dynamic_update["events"] = events_data
        latest_dynamic_update["original_timetable"] = base_timetable
        
        # Save config temporarily for dynamic updater
        temp_config_path = tempfile.mktemp(suffix='.json')
        with open(temp_config_path, 'w') as f:
            json.dump(config_source, f, indent=2)
        
        try:
            # Initialize dynamic updater with existing timetable
            updater = DynamicUpdater(
                config_path=temp_config_path,
                existing_timetable=base_timetable
            )
            
            # Apply events
            updated_result = updater.apply_events(events_data)
            
            # Store updated data
            latest_dynamic_update["data"] = updated_result
            latest_dynamic_update["status"] = "completed"
            latest_dynamic_update["timestamp"] = datetime.now().isoformat()
            logger.info("Dynamic update completed successfully")
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
                
    except Exception as e:
        latest_dynamic_update["status"] = "failed"
        latest_dynamic_update["error"] = str(e)
        logger.error(f"Dynamic update error: {str(e)}")

# =============================================================================
# MAIN ROUTES
# =============================================================================

@app.get("/")
async def home():
    """Basic health check endpoint"""
    return {
        'message': 'Smart Timetable Generator API with Ultra-Fast Cerebras',
        'status': 'running',
        'version': '3.0',
        'llm_backend': 'Cerebras' if USE_CEREBRAS else 'Gemini'
    }

# =============================================================================
# TIMETABLE PARSING ENDPOINTS
# =============================================================================

@app.post("/api/parse-timetable")
async def parse_timetable(
    file: UploadFile = File(...),
    college_name: Optional[str] = Form(None),
    session: Optional[str] = Form(None)
):
    """
    Parse uploaded PDF/Excel timetable file using ultra-fast LLM
    
    - **file**: PDF, XLSX, XLS, or XLSM file
    - **college_name**: Optional college name
    - **session**: Optional session info
    """
    if not llm_extractor:
        raise HTTPException(
            status_code=503,
            detail='LLM extractor is not available. Please check your API key setup.'
        )
    
    try:
        # Validate file
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail='Invalid file format. Allowed: PDF, XLSX, XLS, XLSM, JSON'
            )
        
        # Read file content
        file_content = await file.read()
        
        # Check file size
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail='File too large. Maximum size is 50MB.'
            )
        
        filename = file.filename
        extraction_method = "Cerebras (ultra-fast)" if USE_CEREBRAS else "Gemini"
        logger.info(f"Starting {extraction_method} parsing for file: {filename}")
        
        # Extract timetable data using ultra-fast LLM
        start_time = datetime.now()
        result = llm_extractor.extract_timetable_data(
            file_content=file_content,
            filename=filename,
            college_name=college_name,
            session=session
        )
        end_time = datetime.now()
        extraction_time = (end_time - start_time).total_seconds()
        
        # Note: config no longer cached globally
        
        success_message = f"Timetable parsed successfully using {extraction_method}"
        if USE_CEREBRAS:
            success_message += f" in {extraction_time:.2f}s (ultra-fast!)"
        
        logger.info("Timetable parsing completed successfully")
        
        return {
            'success': True,
            'message': success_message,
            'data': result,
            'extraction_info': {
                **result.get('extraction_info', {}),
                'llm_backend': 'Cerebras' if USE_CEREBRAS else 'Gemini',
                'extraction_time_seconds': extraction_time
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Timetable parsing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f'Failed to parse timetable: {str(e)}'
        )

# =============================================================================
# TIMETABLE GENERATION ENDPOINTS
# =============================================================================

@app.post("/api/generate")
async def generate_timetable(
    request: Optional[Dict[str, Any]] = Body(None)
):
    """
    Generate timetable using configuration and return results directly
    
    Options:
    1. Use JSON configuration from request body
    2. Use default configuration file
    
    Returns the generated timetable data directly without caching
    """
    try:
        config_data = None
        
        # Try to get configuration from request
        if request:
            config_data = request
            logger.info("Using configuration from request body")
        else:
            raise HTTPException(
                status_code=400,
                detail="Config file not found! Please provide configuration in request body."
            )
        
        # Generate timetables synchronously
        try:
            genetic_algo = GeneticAlgorithm(config_data)
            solution = genetic_algo.solve()
            
            if not solution:
                raise HTTPException(
                    status_code=400,
                    detail="No valid solution found!"
                )
            
            # Export to JSON format
            exporter = TimetableExporter(solution, config_data)
            timetable_data = {
                "sections": exporter.export_section_timetables(),
                "faculty": exporter.export_faculty_timetables(),
                "detailed": exporter.export_detailed_timetable(),
                "statistics": exporter.export_statistics()
            }
            
            logger.info("Timetable generation completed successfully")
            
            # Return data directly without caching
            return {
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
                "data": timetable_data
            }
            
        except Exception as e:
            logger.error(f"Timetable generation failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}"
            )
    
    except Exception as e:
        logger.error(f"Failed to generate timetable: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate timetable: {str(e)}"
        )

@app.post("/api/generate-from-parsed")
async def generate_from_parsed():
    """
    DEPRECATED: Generate timetable using cached parsed configuration
    This endpoint no longer works as configs are not cached globally
    Use /api/generate instead with configuration in request body
    """
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Please use /api/generate with configuration in request body."
    )

# =============================================================================
# STATUS & RESULTS ENDPOINTS
# =============================================================================

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "status": "ready",
        "timestamp": datetime.now().isoformat(),
        "llm_backend": "Cerebras" if USE_CEREBRAS else "Gemini",
        "caching_mode": "No global caching - each user gets isolated per-request results",
        "note": "Use /api/generate to generate timetables. Results are returned directly without caching."
    }

@app.get("/api/results")
async def get_all_results():
    """DEPRECATED: Results endpoint"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /api/generate to get results directly."
    )

# =============================================================================
# SECTION TIMETABLE ENDPOINTS
# =============================================================================

@app.get("/api/timetables/sections")
async def get_all_sections():
    """DEPRECATED: Get all section timetables"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /api/generate and access sections from response."
    )

@app.get("/api/timetables/sections/{section_id}")
async def get_single_section(section_id: str):
    """DEPRECATED: Get specific section timetable"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /api/generate and access sections from response."
    )

# =============================================================================
# FACULTY TIMETABLE ENDPOINTS
# =============================================================================

@app.get("/api/timetables/faculty")
async def get_all_faculty():
    """DEPRECATED: Get all faculty timetables"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /api/generate and access faculty from response."
    )

@app.get("/api/timetables/faculty/{faculty_id}")
async def get_single_faculty(faculty_id: str):
    """DEPRECATED: Get specific faculty timetable"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Use /api/generate and access faculty from response."
    )

# =============================================================================
# CONFIGURATION ENDPOINTS
# =============================================================================

@app.get("/api/config/parsed")
async def get_parsed_config():
    """DEPRECATED: Get the most recently parsed configuration"""
    raise HTTPException(
        status_code=410,
        detail="This endpoint is deprecated. Configurations are no longer cached globally. Parse the timetable to get configuration."
    )

@app.post("/api/config/validate")
async def validate_config(config: Dict[str, Any] = Body(...)):
    """
    Validate a timetable configuration
    
    Expected: JSON configuration in request body
    """
    try:
        # Check required fields
        required_fields = [
            'college_info', 'time_slots', 'departments',
            'subjects', 'faculty', 'rooms'
        ]
        
        missing_fields = [field for field in required_fields if field not in config]
        if missing_fields:
            return {
                'valid': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }
        
        # Try to create TimetableData object to validate
        try:
            data = TimetableData(config_dict=config)
            return {
                'valid': True,
                'message': 'Configuration is valid',
                'stats': {
                    'departments': len(data.departments),
                    'sections': len(data.sections),
                    'subjects': len(data.subjects),
                    'faculty': len(data.faculty),
                    'rooms': len(data.rooms),
                    'periods': len(data.period_ids)
                }
            }
        except Exception as validation_error:
            return {
                'valid': False,
                'error': f'Configuration validation failed: {str(validation_error)}'
            }
    
    except Exception as e:
        logger.error(f"Config validation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f'Failed to validate configuration: {str(e)}'
        )

# =============================================================================
# DYNAMIC UPDATE ENDPOINTS
# =============================================================================

@app.post("/api/events/upload")
async def upload_events(
    file: Optional[UploadFile] = File(None),
    events_data: Optional[EventsData] = Body(None)
):
    """Upload events JSON file or JSON data"""
    try:
        events = None
        
        # Check if it's a file upload
        if file:
            if not allowed_file(file.filename):
                raise HTTPException(status_code=400, detail='Invalid file')
            
            file_content = await file.read()
            events = json.loads(file_content.decode('utf-8'))
        
        # Check if it's JSON in request body
        elif events_data:
            events = events_data.dict()
        else:
            raise HTTPException(status_code=400, detail='No events data provided')
        
        # Validate events structure
        if not isinstance(events, dict) or 'events' not in events:
            raise HTTPException(
                status_code=400,
                detail='Invalid events format. Expected {"events": [...]}'
            )
        
        # Store events for later use
        latest_dynamic_update["events"] = events
        
        return {
            'success': True,
            'message': f'Events uploaded successfully. Found {len(events.get("events", []))} events.',
            'events_count': len(events.get('events', []))
        }
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f'Invalid JSON: {str(e)}')
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Events upload error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f'Failed to upload events: {str(e)}'
        )

@app.post("/api/dynamic-update/apply")
async def apply_dynamic_update(
    background_tasks: BackgroundTasks,
    request: Optional[DynamicUpdateRequest] = Body(None)
):
    """Apply dynamic updates to existing timetable"""
    global latest_dynamic_update
    
    if latest_dynamic_update["status"] == "in_progress":
        return {
            "status": "already_running",
            "message": "Dynamic update is already in progress"
        }
    
    try:
        # Get events from request or use previously uploaded
        events_data = None
        use_existing = True
        
        if request:
            if request.events:
                events_data = request.events
            use_existing = request.use_existing_timetable
        
        if not events_data and latest_dynamic_update.get("events"):
            events_data = latest_dynamic_update["events"]
        
        if not events_data:
            raise HTTPException(
                status_code=400,
                detail="No events data available. Please upload events first."
            )
        
        # Check if base timetable exists
        if use_existing and latest_dynamic_update.get("data") is None:
            raise HTTPException(
                status_code=400,
                detail="No base timetable available. Please generate a timetable first."
            )
        
        # Start dynamic update in background
        background_tasks.add_task(
            apply_dynamic_updates_async,
            events_data,
            use_existing
        )
        
        return {
            "status": "started",
            "message": "Dynamic update started"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Apply dynamic update error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start dynamic update: {str(e)}"
        )

@app.get("/api/dynamic-update/status")
async def get_dynamic_update_status():
    """Check the status of the latest dynamic update"""
    return latest_dynamic_update

@app.get("/api/dynamic-update/results")
async def get_dynamic_update_results():
    """Get dynamic update results"""
    if latest_dynamic_update["status"] != "completed":
        raise HTTPException(status_code=404, detail="No completed dynamic update available")
    
    return {
        "updated_timetable": latest_dynamic_update["data"],
        "original_timetable": latest_dynamic_update.get("original_timetable"),
        "events_applied": latest_dynamic_update.get("events"),
        "timestamp": latest_dynamic_update.get("timestamp")
    }

@app.get("/api/dynamic-update/timetables/sections")
async def get_updated_section_timetables():
    """Get updated section timetables"""
    if latest_dynamic_update["status"] != "completed" or not latest_dynamic_update["data"]:
        raise HTTPException(status_code=404, detail="No updated timetables available")
    
    return latest_dynamic_update["data"]["sections"]

@app.get("/api/dynamic-update/timetables/faculty")
async def get_updated_faculty_timetables():
    """Get updated faculty timetables"""
    if latest_dynamic_update["status"] != "completed" or not latest_dynamic_update["data"]:
        raise HTTPException(status_code=404, detail="No updated timetables available")
    
    return latest_dynamic_update["data"]["faculty"]

@app.get("/api/events/current")
async def get_current_events():
    """Get currently uploaded events"""
    if not latest_dynamic_update.get("events"):
        raise HTTPException(status_code=404, detail="No events uploaded")
    
    return latest_dynamic_update["events"]

# =============================================================================
# SYSTEM ENDPOINTS
# =============================================================================

@app.get("/api/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'llm_backend': 'Cerebras' if USE_CEREBRAS else 'Gemini',
        'llm_configured': llm_extractor is not None,
        'cerebras_available': USE_CEREBRAS,
        'gemini_api_available': os.getenv('GEMINI_API_KEY') is not None,
        'caching_mode': 'No global caching - per-request isolation',
        'features': {
            'ultra_fast_extraction': USE_CEREBRAS,
            'llm_extraction': llm_extractor is not None,
            'genetic_algorithm': True,
            'dynamic_updates': True,
            'multi_format_support': True,
            'per_request_isolation': True
        }
    }


# =============================================================================
# RUN THE APPLICATION
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )