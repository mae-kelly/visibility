from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
import structlog
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from api.dependencies.database import get_correlation_service
from api.middleware.auth import get_current_user

logger = structlog.get_logger()
router = APIRouter()

@router.get("/")
async def list_correlations(
    limit: int = 100,
    confidence_threshold: float = 0.7,
    correlation_service = Depends(get_correlation_service),
    current_user = Depends(get_current_user)
):
    """Get list of asset correlations"""
    try:
        # Mock correlation data
        correlations = [
            {
                "correlation_id": f"CORR-{i:06d}",
                "asset1_id": f"ASSET-{i:06d}",
                "asset1_source": "crowdstrike",
                "asset1_hostname": f"server-{i:03d}",
                "asset2_id": f"ASSET-{i+1000:06d}",
                "asset2_source": "cmdb",
                "asset2_hostname": f"srv-{i:03d}",
                "similarity_score": 0.95 - (i * 0.001),
                "correlation_type": "hostname_similarity",
                "correlation_method": "ditto_ml",
                "confidence_score": 0.92 - (i * 0.001),
                "created_at": "2025-01-01T10:00:00Z",
                "validated": i < 50
            }
            for i in range(min(limit, 200))
            if 0.95 - (i * 0.001) >= confidence_threshold
        ]
        
        return {
            "correlations": correlations,
            "total": len(correlations),
            "confidence_threshold": confidence_threshold
        }
        
    except Exception as e:
        logger.error(f"Error listing correlations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve correlations")

@router.post("/validate/{correlation_id}")
async def validate_correlation(
    correlation_id: str,
    is_valid: bool,
    current_user = Depends(get_current_user)
):
    """Validate or reject a correlation"""
    try:
        logger.info(f"Validating correlation {correlation_id}: {is_valid}")
        
        return {
            "correlation_id": correlation_id,
            "validated": is_valid,
            "validated_by": current_user.get("name", "Unknown"),
            "validated_at": "2025-01-01T12:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error validating correlation {correlation_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate correlation")

@router.get("/stats")
async def get_correlation_stats(
    current_user = Depends(get_current_user)
):
    """Get correlation statistics"""
    try:
        return {
            "total_correlations": 1847,
            "validated_correlations": 1203,
            "pending_validation": 644,
            "rejected_correlations": 89,
            "average_confidence": 0.87,
            "correlation_methods": {
                "hostname_similarity": 856,
                "ip_address_match": 432,
                "behavioral_analysis": 348,
                "network_topology": 211
            },
            "source_combinations": {
                "crowdstrike_cmdb": 623,
                "chronicle_splunk": 445,
                "crowdstrike_chronicle": 389,
                "cmdb_splunk": 234,
                "multi_source": 156
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting correlation stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve correlation statistics")
