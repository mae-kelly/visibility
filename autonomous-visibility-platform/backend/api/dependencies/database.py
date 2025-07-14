from functools import lru_cache
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from services.duckdb.unified_asset_service import UnifiedAssetService
from services.ai.visibility_orchestrator import VisibilityOrchestrator
from services.ai.correlation_service import AutonomousCorrelationService
from core.config import settings

@lru_cache()
def get_asset_service() -> UnifiedAssetService:
    """Get unified asset service instance"""
    return UnifiedAssetService(settings.DUCKDB_PATH)

@lru_cache()
def get_visibility_orchestrator() -> VisibilityOrchestrator:
    """Get visibility orchestrator instance"""
    return VisibilityOrchestrator(settings.DUCKDB_PATH, settings.ML_MODEL_PATH)

@lru_cache()
def get_correlation_service() -> AutonomousCorrelationService:
    """Get correlation service instance"""
    return AutonomousCorrelationService(settings.DUCKDB_PATH, settings.ML_MODEL_PATH)
