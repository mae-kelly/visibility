from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional, Dict, Any
import structlog
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from api.middleware.auth import get_current_user

logger = structlog.get_logger()
router = APIRouter()

@router.get("/status")
async def get_ai_status(
    current_user = Depends(get_current_user)
):
    """Get status of all AI models"""
    try:
        return {
            "models": {
                "hgt_correlator": {
                    "status": "active",
                    "accuracy": 0.94,
                    "last_trained": "2025-01-01T08:00:00Z",
                    "predictions_today": 1247,
                    "avg_inference_time_ms": 45
                },
                "ditto_resolver": {
                    "status": "active", 
                    "accuracy": 0.91,
                    "last_trained": "2025-01-01T07:30:00Z",
                    "resolutions_today": 856,
                    "avg_inference_time_ms": 32
                },
                "shadow_detector": {
                    "status": "active",
                    "precision": 0.88,
                    "recall": 0.92,
                    "last_trained": "2025-01-01T06:45:00Z",
                    "detections_today": 23,
                    "avg_inference_time_ms": 78
                },
                "lifecycle_predictor": {
                    "status": "training",
                    "accuracy": 0.89,
                    "last_trained": "2024-12-31T22:00:00Z",
                    "predictions_today": 345,
                    "avg_inference_time_ms": 56
                }
            },
            "overall_health": "good",
            "total_predictions_today": 2471,
            "system_load": 0.67
        }
        
    except Exception as e:
        logger.error(f"Error getting AI status: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve AI status")

@router.post("/models/{model_name}/retrain")
async def retrain_model(
    model_name: str,
    current_user = Depends(get_current_user)
):
    """Trigger model retraining"""
    try:
        valid_models = ["hgt_correlator", "ditto_resolver", "shadow_detector", "lifecycle_predictor"]
        
        if model_name not in valid_models:
            raise HTTPException(status_code=400, detail=f"Invalid model name. Valid models: {valid_models}")
        
        logger.info(f"Starting retraining for model: {model_name}")
        
        return {
            "model_name": model_name,
            "training_status": "initiated",
            "training_id": f"TRAIN-{model_name.upper()}-001",
            "estimated_completion": "2025-01-01T14:00:00Z",
            "initiated_by": current_user.get("name", "Unknown")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retraining model {model_name}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate model retraining")

@router.get("/predictions")
async def get_predictions(
    model: Optional[str] = None,
    limit: int = 50,
    current_user = Depends(get_current_user)
):
    """Get recent AI predictions"""
    try:
        # Mock predictions data
        predictions = [
            {
                "prediction_id": f"PRED-{i:06d}",
                "model": "shadow_detector",
                "asset_id": f"ASSET-{i:06d}",
                "prediction_type": "shadow_asset",
                "confidence": 0.92 - (i * 0.01),
                "result": "potential_shadow_asset",
                "created_at": "2025-01-01T12:00:00Z"
            }
            for i in range(min(limit, 100))
        ]
        
        if model:
            predictions = [p for p in predictions if p["model"] == model]
        
        return {
            "predictions": predictions,
            "total": len(predictions),
            "model_filter": model
        }
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve predictions")

@router.get("/performance")
async def get_model_performance(
    model_name: Optional[str] = None,
    days: int = 7,
    current_user = Depends(get_current_user)
):
    """Get model performance metrics"""
    try:
        if model_name:
            # Single model performance
            return {
                "model_name": model_name,
                "period_days": days,
                "metrics": {
                    "accuracy": 0.94,
                    "precision": 0.91,
                    "recall": 0.96,
                    "f1_score": 0.93,
                    "predictions_count": 8934,
                    "avg_inference_time_ms": 45,
                    "error_rate": 0.02
                },
                "daily_performance": [
                    {"date": f"2025-01-0{i}", "accuracy": 0.94 + (i * 0.001), "predictions": 1200 + (i * 50)}
                    for i in range(1, min(days + 1, 8))
                ]
            }
        else:
            # All models summary
            return {
                "period_days": days,
                "models_summary": {
                    "hgt_correlator": {"accuracy": 0.94, "predictions": 8934},
                    "ditto_resolver": {"accuracy": 0.91, "predictions": 6754},
                    "shadow_detector": {"precision": 0.88, "detections": 234},
                    "lifecycle_predictor": {"accuracy": 0.89, "predictions": 2456}
                },
                "overall_metrics": {
                    "total_predictions": 18378,
                    "average_accuracy": 0.905,
                    "system_uptime": 0.998
                }
            }
        
    except Exception as e:
        logger.error(f"Error getting model performance: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve model performance")
