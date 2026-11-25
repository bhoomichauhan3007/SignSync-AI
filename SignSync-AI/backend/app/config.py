from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "SignSync AI Backend"
    database_url: str = "sqlite:///./signsync.db"
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    cors_origins: str = "http://localhost:3000,http://localhost:5173"

    # AI integration
    ai_mode: str = "mock"  # "mock" or "real"
    confidence_threshold: float = 0.70
    ws_stability_frames: int = 5

    # Misc
    login_rate_limit: int = 10  # login attempts per minute per IP
    max_audio_bytes: int = 10 * 1024 * 1024
    admin_email: str = "admin@signsync.dev"
    admin_password: str = "admin12345"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
