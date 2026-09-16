import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Policy-Aware Multi-Agent RAG Claim Decision Engine"
    app_version: str = "1.0.0"
    gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "")
    pdf_path: str = "data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    chroma_db_dir: str = "chroma_db"
    port: int = 8000
    host: str = "0.0.0.0"

    class Config:
        env_file = ".env"

settings = Settings()
