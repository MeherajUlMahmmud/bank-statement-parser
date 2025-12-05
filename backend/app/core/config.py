from typing import List
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application Settings
    APP_NAME: str = "Bank Statement Parser"
    VERSION: str = "1.0.0"
    SECRET_KEY: str = "change-me-in-production"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 5000

    # CORS Settings
    CORS_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    # Database Settings
    DATABASE_URL: str = "sqlite+aiosqlite:///./database.db"

    # File Upload Settings
    UPLOAD_DIR: str = "./uploads"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: str = ".pdf"

    # Groq API Settings
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.2-90b-vision-preview"
    GROQ_TEMPERATURE: float = 0.1
    GROQ_MAX_TOKENS: int = 8192
    GROQ_REQUEST_TIMEOUT: int = 120

    # OlmOCR Settings (for OCR extraction)
    OLMOCR_BASE_URL: str = "http://localhost:8000"
    OLMOCR_REQUEST_TIMEOUT: int = 300
    OLMOCR_MAX_RETRIES: int = 3
    OLMOCR_RETRY_DELAY: float = 1.0

    # Processing Settings
    CLEANUP_TEMP_FILES: bool = True
    ENABLE_BACKGROUND_PROCESSING: bool = True

    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"
    LOG_MAX_BYTES: int = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT: int = 30
    LOG_FORMAT: str = "json"  # json or text

    # Confidence Scoring Settings
    CONFIDENCE_THRESHOLD: float = 0.70
    HEURISTIC_WEIGHT: float = 0.6
    VLM_WEIGHT: float = 0.4

    # PII Settings
    MASK_PII: bool = True
    PII_MASK_CHAR: str = "X"
    PII_SHOW_LAST: int = 4

    # PDF Processing Settings
    PDF_DPI: int = 300
    PDF_IMAGE_FORMAT: str = "PNG"

    # YOLO Settings (if enabled)
    YOLO_MODEL_PATH: str = ""
    YOLO_CONFIDENCE_THRESHOLD: float = 0.5
    YOLO_SAVE_ANNOTATED_IMAGE: bool = False
    YOLO_SAVE_CROPPED_IMAGE: bool = False
    YOLO_PROJECT_PATH: str = "./yolo_output"

    # PaddleOCR Settings (if enabled)
    USE_GPU: bool = False
    DET_MODEL_DIR: str = ""
    REC_MODEL_DIR: str = ""
    CLS_MODEL_DIR: str = ""
    USE_ANGLE_CLS: bool = True
    DROP_SCORE: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if isinstance(self.CORS_ORIGINS, str):
            return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        return []

    @property
    def upload_path(self) -> Path:
        """Get upload directory as Path object."""
        path = Path(self.UPLOAD_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_path(self) -> Path:
        """Get log directory as Path object."""
        path = Path(self.LOG_DIR)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def allowed_extensions_list(self) -> List[str]:
        """Parse allowed file extensions from comma-separated string."""
        if isinstance(self.ALLOWED_EXTENSIONS, str):
            return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",") if ext.strip()]
        return [".pdf"]


settings = Settings()
