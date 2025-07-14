from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
import structlog
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from api.schemas.assets import Asset, AssetListResponse, AssetSearchRequest, AssetCreate, AssetUpdate
from api.dependencies.database import get_asset_service
from api.middleware.auth import get_current_user
from services.duckdb.unified_asset_service import UnifiedAssetService

logger = structlog.get_logger()
router = APIRouter()

@router.get("/", response_model=AssetListResponse)
async def list_assets(
    page: int = Query(1, ge=1),
    size: int = Query(50, ge=1, le=1000),
    asset_class: Optional[str] = Query(None),
    environment: Optional[str] = Query(None),
    min_visibility: Optional[float] = Query(None, ge=0, le=100),
    max_visibility: Optional[float] = Query(None, ge=0, le=100),
    db: UnifiedAssetService = Depends(get_asset_service),
    current_user = Depends(get_current_user)
):
    """Get paginated list of assets with optional filtering"""
    try:
        # Build filter conditions
        filters = {}
        if asset_class:
            filters['asset_class'] = asset_class
        if environment:
            filters['environment'] = environment
        if min_visibility is not None:
            filters['min_visibility_score'] = min_visibility
        if max_visibility is not None:
            filters['max_visibility_score'] = max_visibility
        
        # Get coverage analysis for total count
        coverage = db.get_coverage_analysis()
        total_assets = coverage['total_assets']
        
        # For now, return mock paginated response
        # In a real implementation, you'd query the database with filters
        mock_assets = [
            {
                "asset_id": f"ASSET-{i:06d}",
                "primary_hostname": f"server-{i:03d}",
                "all_hostnames": [f"server-{i:03d}", f"srv-{i:03d}"],
                "primary_ip": f"10.0.{i//100}.{i%100}",
                "all_ips": [f"10.0.{i//100}.{i%100}"],
                "asset_class": "server",
                "environment": "production",
                "location": "DC1",
                "region": "us-east-1",
                "country": "US",
                "source_systems": ["crowdstrike", "cmdb"],
                "visibility_score": 85.0 + (i % 15),
                "risk_score": 30 + (i % 70),
                "last_seen": "2025-01-01T12:00:00Z",
                "created_at": "2025-01-01T10:00:00Z",
                "updated_at": "2025-01-01T12:00:00Z",
                "correlation_confidence": 0.95
            }
            for i in range((page-1) * size, min(page * size, total_assets))
        ]
        
        return AssetListResponse(
            assets=mock_assets,
            total=total_assets,
            page=page,
            size=size,
            has_more=page * size < total_assets
        )
        
    except Exception as e:
        logger.error(f"Error listing assets: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve assets")

@router.get("/{asset_id}", response_model=Asset)
async def get_asset(
    asset_id: str,
    db: UnifiedAssetService = Depends(get_asset_service),
    current_user = Depends(get_current_user)
):
    """Get detailed information about a specific asset"""
    try:
        # Mock asset response
        mock_asset = {
            "asset_id": asset_id,
            "primary_hostname": "web-server-01",
            "all_hostnames": ["web-server-01", "web01", "webserver-prod"],
            "primary_ip": "10.0.1.100",
            "all_ips": ["10.0.1.100", "192.168.1.100"],
            "asset_class": "server",
            "environment": "production",
            "location": "DC1-Rack-A-Unit-15",
            "region": "us-east-1",
            "country": "US",
            "source_systems": ["crowdstrike", "cmdb", "splunk", "chronicle"],
            "visibility_score": 95.0,
            "risk_score": 25,
            "last_seen": "2025-01-01T12:00:00Z",
            "created_at": "2025-01-01T10:00:00Z",
            "updated_at": "2025-01-01T12:00:00Z",
            "correlation_confidence": 0.98
        }
        
        return mock_asset
        
    except Exception as e:
        logger.error(f"Error retrieving asset {asset_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve asset")

@router.post("/search", response_model=AssetListResponse)
async def search_assets(
    search_request: AssetSearchRequest,
    db: UnifiedAssetService = Depends(get_asset_service),
    current_user = Depends(get_current_user)
):
    """Advanced asset search with complex filtering"""
    try:
        logger.info(f"Asset search request: {search_request}")
        
        # Mock search results
        mock_results = [
            {
                "asset_id": f"SEARCH-{i:03d}",
                "primary_hostname": f"match-{i}",
                "all_hostnames": [f"match-{i}"],
                "primary_ip": f"10.1.{i}.1",
                "all_ips": [f"10.1.{i}.1"],
                "asset_class": search_request.asset_class or "server",
                "environment": search_request.environment or "production",
                "location": "DC1",
                "region": "us-east-1",
                "country": "US",
                "source_systems": ["crowdstrike", "cmdb"],
                "visibility_score": 75.0 + (i * 5),
                "risk_score": 40 + (i * 3),
                "last_seen": "2025-01-01T12:00:00Z",
                "created_at": "2025-01-01T10:00:00Z",
                "updated_at": "2025-01-01T12:00:00Z",
                "correlation_confidence": 0.85
            }
            for i in range(1, min(search_request.size + 1, 20))
        ]
        
        return AssetListResponse(
            assets=mock_results,
            total=len(mock_results),
            page=search_request.page,
            size=search_request.size,
            has_more=False
        )
        
    except Exception as e:
        logger.error(f"Error searching assets: {e}")
        raise HTTPException(status_code=500, detail="Asset search failed")

@router.get("/{asset_id}/correlations")
async def get_asset_correlations(
    asset_id: str,
    db: UnifiedAssetService = Depends(get_asset_service),
    current_user = Depends(get_current_user)
):
    """Get all correlations for a specific asset"""
    try:
        mock_correlations = [
            {
                "correlation_id": f"CORR-{i:04d}",
                "related_asset_id": f"ASSET-{i:06d}",
                "correlation_type": "hostname_match",
                "confidence_score": 0.95 - (i * 0.05),
                "source_systems": ["crowdstrike", "cmdb"],
                "created_at": "2025-01-01T10:00:00Z"
            }
            for i in range(1, 6)
        ]
        
        return {
            "asset_id": asset_id,
            "correlations": mock_correlations,
            "total_correlations": len(mock_correlations)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving correlations for asset {asset_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve correlations")
