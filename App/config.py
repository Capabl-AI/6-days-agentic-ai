import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # App Settings
    PORT: int = Field(default=8080, validation_alias="PORT")

    # Huggingface settings
    HF_MODEL_NAME: str = Field(validation_alias="HF_MODEL_NAME")
    HF_AUTH_TOKEN: str = Field(validation_alias="HF_AUTH_TOKEN")

    # Groq Settings
    GROQ_API_KEY: str = Field(validation_alias="GROQ_API_KEY")
    GROQ_MODEL_NAME: str = Field(validation_alias="GROQ_MODEL_NAME")

    # Langsmith Settings
    LANGSMITH_TRACING: str = Field(validation_alias="LANGSMITH_TRACING")
    LANGSMITH_ENDPOINT: str = Field(validation_alias="LANGSMITH_ENDPOINT")
    LANGSMITH_API_KEY: str = Field(validation_alias="LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = Field(validation_alias="LANGSMITH_PROJECT")

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

settings = Settings()

os.environ["LANGSMITH_TRACING"] = settings.LANGSMITH_TRACING
os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT
os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT