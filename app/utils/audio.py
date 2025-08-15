import shutil
import uuid
from fastapi import UploadFile, HTTPException
from app.core.config import settings
import re


def save_upload_file(upload_file: UploadFile, destination_folder: str, ext=".wav") -> str:
    if upload_file.content_type not in settings.allowed_mime_types:
        print(upload_file.content_type)
        raise HTTPException(status_code=400, detail="Invalid audio format")

    filename = f"{str(uuid.uuid4())}"+ext
    file_path = f"{destination_folder}/{filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)

    return file_path

def parse_analysis_result(text: str):
    lines = text.strip().splitlines()
    result = {}
    current_key = None

    for line in lines:
        line = line.strip()
        if line.startswith("/noutput") or line.startswith("/nuploads"):
            filename = line.split("for")[-1].strip(": ")
            current_key = "original_audio" if "uploads" in filename else "cleaned_audio"
            result[current_key] = {"file": filename, "predictions": {}}
        elif ":" in line and current_key:
            match = re.match(r'"?(.*?)"?\s*:\s*([0-9.]+)', line)
            if match:
                label, value = match.groups()
                result[current_key]["predictions"][label.strip()] = float(value)

    return result
