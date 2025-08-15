from pydantic import BaseModel

class CloneResponse(BaseModel):
    message: str
    cloned_audio_url: str

class NoiseCleanResponse(BaseModel):
    message: str
    clean_audio_url: str