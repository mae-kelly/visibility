from fastapi import FastAPI
import uvicorn
from datetime import datetime

app = FastAPI(
    title="Autonomous Visibility Platform",
    version="1.0.0",
    docs_url="/api/docs"
)

@app.get("/")
async def root():
    return {"message": "Autonomous Visibility Platform API", "status": "running"}

@app.get("/api/health")
async def health():
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }

@app.get("/api/v1/assets")
async def get_assets():
    return {
        "assets": [
            {"id": "asset-1", "hostname": "server-01", "ip": "10.0.1.1", "visibility": 85, "environment": "production"},
            {"id": "asset-2", "hostname": "server-02", "ip": "10.0.1.2", "visibility": 92, "environment": "production"},
            {"id": "asset-3", "hostname": "server-03", "ip": "10.0.1.3", "visibility": 78, "environment": "staging"},
            {"id": "asset-4", "hostname": "web-srv-01", "ip": "10.0.1.10", "visibility": 95, "environment": "production"},
            {"id": "asset-5", "hostname": "db-srv-01", "ip": "10.0.1.20", "visibility": 88, "environment": "production"}
        ],
        "total": 5
    }

@app.get("/api/v1/visibility/dashboard")
async def get_dashboard():
    return {
        "total_assets": 200,
        "visibility_percentage": 85.3,
        "high_visibility": 156,
        "gaps": 23,
        "status": "good"
    }

@app.get("/api/v1/gaps")
async def get_gaps():
    gaps = []
    for i in range(1, 24):
        gaps.append({
            "asset_id": f"ASSET-{i:06d}",
            "hostname": f"server-{i:03d}",
            "visibility_score": 45.0 + (i % 30),
            "gap_severity": "High" if i % 3 == 0 else "Medium",
            "missing_sources": [
                "CrowdStrike agent deployment needed",
                "Chronicle logging configuration required"
            ]
        })
    
    return {
        "gaps": gaps,
        "total": len(gaps)
    }

@app.get("/api/v1/ai/status")
async def get_ai_status():
    return {
        "models": {
            "asset_correlator": {
                "status": "active",
                "accuracy": 0.94,
                "predictions_today": 1247
            },
            "entity_resolver": {
                "status": "active",
                "accuracy": 0.91,
                "resolutions_today": 856
            },
            "shadow_detector": {
                "status": "active",
                "precision": 0.88,
                "detections_today": 23
            }
        },
        "overall_health": "good",
        "total_predictions_today": 2126
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
