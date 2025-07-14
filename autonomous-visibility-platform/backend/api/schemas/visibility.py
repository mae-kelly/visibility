from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class VisibilityStatus(str, Enum):
    EXCELLENT = "excellent"  # 90-100%
    GOOD = "good"           # 75-89%
    FAIR = "fair"           # 50-74%
    POOR = "poor"           # <50%

class OrchestrationPhase(str, Enum):
    INITIALIZATION = "initialization"
    DISCOVERY = "discovery"
    CORRELATION = "correlation"
    GAP_ANALYSIS = "gap_analysis"
    REMEDIATION = "remediation"
    OPTIMIZATION = "optimization"
    COMPLETED = "completed"

class VisibilityMetrics(BaseModel):
    total_assets: int
    high_visibility_assets: int
    medium_visibility_assets: int
    low_visibility_assets: int
    average_visibility_score: float = Field(..., ge=0, le=100)
    visibility_percentage: float = Field(..., ge=0, le=100)
    unique_source_systems: int

class CoverageAnalysis(BaseModel):
    source_system: str
    assets_covered: int
    total_assets: int
    coverage_percentage: float = Field(..., ge=0, le=100)
    average_confidence: float = Field(..., ge=0, le=1)

class VisibilityDashboard(BaseModel):
    metrics: VisibilityMetrics
    status: VisibilityStatus
    coverage_by_source: List[CoverageAnalysis]
    recent_discoveries: int
    pending_correlations: int
    active_gaps: int
    last_updated: datetime

class OrchestrationRequest(BaseModel):
    target_visibility: float = Field(100.0, ge=80, le=100)
    enable_auto_remediation: bool = True
    enable_shadow_discovery: bool = True
    max_execution_time_minutes: int = Field(60, ge=10, le=480)

class OrchestrationStatus(BaseModel):
    orchestration_id: str
    phase: OrchestrationPhase
    progress_percentage: float = Field(..., ge=0, le=100)
    current_visibility: float = Field(..., ge=0, le=100)
    target_visibility: float = Field(..., ge=0, le=100)
    actions_completed: int
    actions_remaining: int
    estimated_completion: Optional[datetime] = None
    is_running: bool
    last_updated: datetime

class OrchestrationResult(BaseModel):
    orchestration_id: str
    success: bool
    final_visibility_score: float = Field(..., ge=0, le=100)
    actions_taken: List[str]
    gaps_resolved: int
    assets_discovered: int
    correlations_created: int
    execution_time_seconds: float
    completion_time: datetime
