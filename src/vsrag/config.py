from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings. Each field can be overridden by an environment variable or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # 127.0.0.1, not localhost: on Windows, localhost tries IPv6 first, and Docker's port
    # only answers on IPv4, so the IPv6 attempt hangs. The timeout turns any future hang
    # into an error.
    database_url: str = "postgresql://vsrag:vsrag@127.0.0.1:5432/vsrag?connect_timeout=10"
    test_database_url: str = "postgresql://vsrag:vsrag@127.0.0.1:5432/vsrag_test?connect_timeout=10"
    ollama_url: str = "http://localhost:11434"
    answer_model: str = "qwen3:8b"
    judge_model: str = "llama3.1:8b"


settings = Settings()
