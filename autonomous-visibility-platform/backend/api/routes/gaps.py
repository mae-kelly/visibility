from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional
import structlog
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from api.dependencies.database import get_asset_service
from api.middleware.auth import get_current_user
from services.duckdb.unified_asset_service import UnifiedAssetService

logger = structlog.get_logger()
router = APIRouter()

@router.get("/")
async def list_gaps(
    severity: Optional[str] = None,
    limit: int = 100,
    db: UnifiedAssetService = Depends(get_asset_service),
    current_user = Depends(get_current_user)
):
    """Get list of visibility gaps"""
    try:
        gaps = db.get_gap_analysis()
        
        if severity:
            gaps = [gap for gap in gaps if gap['gap_severity'].lower() == severity.lower()]
        
        gaps = gaps[:limit]
        
        return {
            "gaps": gaps,
            "total": len(gaps),
            "severity_filter": severity
        }
        
    except Exception as e:
        logger.error(f"Error listing gaps: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve gaps")

@router.get("/summary")
async def get_gap_summary(
    db: UnifiedAssetService = Depends(get_asset_service),
    current_user = Depends(get_current_user)
):
    """Get gap analysis summary"""
    try:
        gaps = db.get_gap_analysis()
        
        summary = {
            "total_gaps": len(gaps),
            "critical_gaps": len([g for g in gaps if g['gap_severity'] == 'High']),
            "medium_gaps": len([g for g in gaps if g['gap_severity'] == 'Medium']),
            "low_gaps": len([g for g in gaps if g['gap_severity'] == 'Low']),
            "gaps_by_source": {},
            "gaps_by_environment": {},
            "remediation_estimates": {
                "immediate_action_required": 0,
                "scheduled_remediation": 0,
                "low_priority": 0
            }
        }
        
        # Analyze gaps by missing sources
        for gap in gaps:
            for missing_source in gap['missing_sources']:
                source_name = missing_source.split()[0]  # Extract source name
                summary["gaps_by_source"][source_name] = summary["gaps_by_source"].get(source_name, 0) + 1
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting gap summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve gap summary")

@router.post("/remediate/{asset_id}")
async def remediate_gap(
    asset_id: str,
    action_type: str,
    current_user = Depends(get_current_user)
):
    """Trigger automated gap remediation for an asset"""
    try:
        logger.info(f"Starting remediation for asset {asset_id}: {action_type}")
        
        # Mock remediation response
        return {
            "asset_id": asset_id,
            "action_type": action_type,
            "status": "initiated",
            "estimated_completion": "2025-01-01T12:30:00Z",
            "remediation_id": f"REM-{asset_id}-{action_type.upper()}",
            "initiated_by": current_user.get("name", "Unknown")
        }
        
    except Exception as e:
        logger.error(f"Error remediating gap for asset {asset_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to initiate remediation")
