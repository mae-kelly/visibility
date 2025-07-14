from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Autonomous Visibility Platform"
    
    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "https://localhost:3000"
    ]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DUCKDB_PATH: str = "./data/duckdb/security_visibility.db"
    
    # ML Models
    ML_MODEL_PATH: str = "./ml_engine/models"
    
    # Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "development-secret-key")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Azure AD
    AZURE_CLIENT_ID: str = os.getenv("CLIENT_ID", "")
    AZURE_CLIENT_SECRET: str = os.getenv("CLIENT_SECRET", "")
    AZURE_TENANT_ID: str = os.getenv("AUTHORITY", "").split("/")[-1] if os.getenv("AUTHORITY") else ""
    
    # External APIs
    CHRONICLE_API_KEY: str = os.getenv("CHRONICLE_API_KEY", "")
    CHRONICLE_ENDPOINT: str = os.getenv("CHRONICLE_ENDPOINT", "")
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Kafka
    KAFKA_BROKERS: str = os.getenv("KAFKA_BROKERS", "localhost:9092")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    
    class Config:
        env_file = ".env"

settings = Settings()
