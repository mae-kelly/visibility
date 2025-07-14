from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from services.duckdb.real_visibility_service import RealVisibilityService

app = FastAPI(
    title="Real AO1 Visibility Platform",
    version="1.0.0",
    docs_url="/api/docs"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Initialize the real database service
# Update this path to your actual DuckDB file
DB_PATH = "/path/to/your/actual/database.duckdb"  # <-- UPDATE THIS
visibility_service = RealVisibilityService(DB_PATH)

@app.get("/")
async def root():
    return {"message": "Real AO1 Visibility Platform", "status": "online"}

@app.get("/api/health")
async def health():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "database": "connected"
    }

@app.get("/api/v1/assets/all")
async def get_all_assets():
    """Get all unified assets from your actual database"""
    try:
        assets = visibility_service.correlate_assets_simple()
        return {
            "assets": assets,
            "total": len(assets)
        }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@app.get("/api/v1/coverage/metrics")
async def get_coverage_metrics():
    """Get real coverage metrics from your data"""
    try:
        metrics = visibility_service.get_coverage_metrics()
        return metrics
    except Exception as e:
        return {"error": f"Metrics calculation failed: {str(e)}"}

@app.get("/api/v1/gaps/list")
async def get_gaps():
    """Get actionable list of gaps to fix"""
    try:
        gaps = visibility_service.get_gaps_to_fix()
        return {
            "gaps": gaps,
            "total": len(gaps)
        }
    except Exception as e:
        return {"error": f"Gap analysis failed: {str(e)}"}

@app.get("/api/v1/gaps/by-source")
async def get_gaps_by_source():
    """Get gaps broken down by source system"""
    try:
        gaps = visibility_service.get_source_specific_gaps()
        return gaps
    except Exception as e:
        return {"error": f"Source gap analysis failed: {str(e)}"}

@app.get("/api/v1/sources/raw")
async def get_raw_sources():
    """Get raw data from each source system"""
    try:
        sources = visibility_service.get_all_unique_assets()
        result = {}
        for source_name, df in sources.items():
            result[source_name] = {
                "count": len(df),
                "sample_data": df.head(5).to_dict('records') if not df.empty else []
            }
        return result
    except Exception as e:
        return {"error": f"Raw data retrieval failed: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
