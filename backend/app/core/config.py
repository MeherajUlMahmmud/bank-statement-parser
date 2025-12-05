from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SECRET_KEY: str = "MyApp"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = ""
    MODEL_TEMP: float = 0.7
    MODEL_MAX_TOKEN: int = 1024
    DATABASE_URL: str = ""
    # cors_origins: List[str] = []
    CLEANUP_TEMP_FILES: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def cors_origins(self) -> list[str]:
        origins = self.cors.origins
        if isinstance(origins, str):
            return [i.strip() for i in origins.split(",")]
        return origins


settings = Settings()
# print(settings.model_dump_json(indent=2))
