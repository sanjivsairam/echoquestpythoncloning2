import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from huggingface_hub import snapshot_download

from app.api.v1.endpoints.dubbing import router as dubbing_router
from app.cron import cleanupjob
from app.middleware.audit_logger_db import AuditLoggerDBMiddleware

#os.environ["TRANSFORMERS_OFFLINE"] = "1"



cleanupjob.start_cron_job()


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = "/models/faster-whisper-large-v2"
    if not os.path.exists(model_dir):
        print("📦 Downloading Hugging Face model...")
        snapshot_download(
            repo_id="guillaumekln/faster-whisper-large-v2",
            local_dir=model_dir,
            local_dir_use_symlinks=False
        )
    else:
        print("✅ Hugging Face model already present.")

    yield  # App is now running
    # Cleanup code (if needed) can go here


app = FastAPI(title="Voice-Cloning-Service2",lifespan=lifespan)

#app = FastAPI(title="Voice-Cloning-Service2")
app.add_middleware(AuditLoggerDBMiddleware)
app.include_router(dubbing_router, prefix="/api")


@app.get("/")
def read_root():
    return {"message": "Voice Cloning Service 2 running"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)