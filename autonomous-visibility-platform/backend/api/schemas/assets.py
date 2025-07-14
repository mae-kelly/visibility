from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class AssetClass(str, Enum):
    SERVER = "server"
    WORKSTATION = "workstation"
    NETWORK_DEVICE = "network_device"
    MOBILE = "mobile"
    CLOUD_INSTANCE = "cloud_instance"
    UNKNOWN = "unknown"

class Environment(str, Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    TEST = "test"
    UNKNOWN = "unknown"

class AssetBase(BaseModel):
    primary_hostname: Optional[str] = None
    all_hostnames: List[str] = []
    primary_ip: Optional[str] = None
    all_ips: List[str] = []
    asset_class: AssetClass = AssetClass.UNKNOWN
    environment: Environment = Environment.UNKNOWN
    location: Optional[str] = None
    region: Optional[str] = None
    country: str = "US"

class AssetCreate(AssetBase):
    discovery_sources: List[str] = []

class AssetUpdate(AssetBase):
    visibility_score: Optional[float] = Field(None, ge=0, le=100)
    risk_score: Optional[int] = Field(None, ge=0, le=100)

class Asset(AssetBase):
    asset_id: str
    source_systems: List[str]
    visibility_score: float = Field(..., ge=0, le=100)
    risk_score: int = Field(..., ge=0, le=100)
    last_seen: datetime
    created_at: datetime
    updated_at: datetime
    correlation_confidence: Optional[float] = Field(None, ge=0, le=1)
    
    class Config:
        from_attributes = True

class AssetListResponse(BaseModel):
    assets: List[Asset]
    total: int
    page: int
    size: int
    has_more: bool

class AssetSearchRequest(BaseModel):
    query: Optional[str] = None
    asset_class: Optional[AssetClass] = None
    environment: Optional[Environment] = None
    min_visibility_score: Optional[float] = Field(None, ge=0, le=100)
    max_visibility_score: Optional[float] = Field(None, ge=0, le=100)
    source_systems: Optional[List[str]] = None
    page: int = Field(1, ge=1)
    size: int = Field(50, ge=1, le=1000)
