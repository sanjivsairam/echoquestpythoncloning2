from typing import ClassVar, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Voice-Cloning-Microservice2"
    openvoice_model_path: str = "./models"

    allowed_mime_types: ClassVar[List[str]] = [
        "audio/wav",
        "audio/x-wav",
        "audio/mpeg",
        "audio/mp3",
        "audio/x-mp3",
        "audio/mp4",
        "audio/m4a",
        "audio/x-m4a",
        "audio/aac",
        "audio/x-aac",
        "audio/flac",
        "audio/x-flac",
        "audio/ogg",
        "application/ogg",
        "audio/x-ms-wma",
        "audio/webm",
        "video/webm",
        "audio/webm;codecs=opus",
        "audio/webm; codecs=opus",
        "video/quicktime",
        "video/x-matroska",
        "video/x-msvideo",
        "video/mp4",
        "video/x-ms-wmv",
        "video/3gpp",
        "video/MP2T",
        "application/x-mpegURL",
        "video/x-flv"
    ]


settings = Settings()
