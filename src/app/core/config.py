"""Configuration management for the multi-agent RAG system."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # OpenAI
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_model_name: str = "gpt-4o-mini"
    openai_embedding_model_name: str = "text-embedding-3-large"

    # Pinecone
    pinecone_api_key: str = Field(alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(alias="PINECONE_INDEX_NAME")

    # Retrieval
    retrieval_k: int = 4