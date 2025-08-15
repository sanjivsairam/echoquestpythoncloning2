# Production-ready FastAPI for OpenVoice Voice Cloning

import os
import traceback

import torch
import shutil
import uuid
from pathlib import Path
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, APIRouter
from fastapi.responses import FileResponse, JSONResponse
import subprocess

#from app.core.config import settings
from app.utils.audio import save_upload_file

# Suppress SSL CA bundle warnings for torch hub
os.environ['CURL_CA_BUNDLE'] = ''

from melo.api import TTS
import nltk


import sys
from pathlib import Path

# Update this path to wherever the OpenVoice repo folder is located
OPENVOICE_REPO_PATH = Path("/").resolve()

# Add to sys.path for dynamic imports

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent.parent  # This reaches echoquestcloning2/
OPENVOICE_DIR = BASE_DIR / "OpenVoice"
print(OPENVOICE_DIR)
# Add OpenVoice directory to sys.path
sys.path.append(str(OPENVOICE_DIR))
from openvoice.api import ToneColorConverter
from openvoice import se_extractor
router = APIRouter()

nltk.download('averaged_perceptron_tagger_eng')

app = FastAPI()

# Directories
UPLOAD_DIR = Path("uploads")
TEMP_UPLOAD_DIR = Path("temp_uploads")
OUTPUT_DIR = Path("cloned")
CHECKPOINT_DIR = Path("checkpoints_v2")
BASE_SPEAKER_DIR = CHECKPOINT_DIR / "base_speakers" / "ses"
CONFIG_PATH = CHECKPOINT_DIR / "converter" / "config.json"
CKPT_PATH = CHECKPOINT_DIR / "converter" / "checkpoint.pth"

print(UPLOAD_DIR.parent)
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Setup model
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
#DEVICE = "cuda:0" if torch.cuda.is_available() else torch.device("cpu")
tone_color_converter = ToneColorConverter(str(CONFIG_PATH), device=DEVICE)
tone_color_converter.load_ckpt(str(CKPT_PATH))

#ALLOWED_FORMATS = [".wav", ".mp3"]
ALLOWED_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm",".mov",".mp4",".avi",".mkv"]

@router.post("/clone-voice")
async def clone_voice(
    reference_audio: UploadFile = File(...),
    input_text: str = Form(...),
    language: str = Form("EN"),
    speed: float = Form(1.0)
):
    ext = Path(reference_audio.filename).suffix.lower()
    #if not reference_audio.filename.endswith(".wav"):
    if ext not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail="Provided audio format not supported !!")

   # ref_id = uuid.uuid4().hex[:8]
   # ref_path = UPLOAD_DIR / f"ref_{ref_id}.wav"
    #with open(ref_path, "wb") as f:
    #    shutil.copyfileobj(reference_audio.file, f)
    input_path = ""
    output_path = OUTPUT_DIR / f"cloned_{Path(input_path).name}"
    try:
        if ext != ".wav":
            temp_input_path = save_upload_file(reference_audio, "temp_uploads",ext)
            convert_to_wav(Path(temp_input_path),UPLOAD_DIR / (Path(temp_input_path).name+".wav"))
            input_path = UPLOAD_DIR / (Path(temp_input_path).name+".wav")
        else:
            input_path = save_upload_file(reference_audio, "uploads",ext)

        output_path = OUTPUT_DIR / f"cloned_{Path(input_path).name}"

        target_se, _ = se_extractor.get_se(str(input_path), tone_color_converter, vad=True)
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Speaker embedding extraction failed: {str(e)}")

    language = custom_lang = language.upper()
    if language in ["EN-US","EN-BR","EN_INDIA","EN-AU"]:
        language = "EN"

    model = TTS(language=language, device=DEVICE)
    speaker_ids = model.hps.data.spk2id

    #src_path = OUTPUT_DIR / f"tmp_{ref_id}.wav"
    results = {}
    result_dict = {}
    for speaker_key in speaker_ids.keys():
        if speaker_key != custom_lang:
            continue

        speaker_id = speaker_ids[speaker_key]
        speaker_key_fmt = speaker_key.lower().replace('_', '-')

        try:
            source_se_path = BASE_SPEAKER_DIR / f"{speaker_key_fmt}.pth"
            source_se = torch.load(source_se_path, map_location=DEVICE)
            model.tts_to_file(input_text, speaker_id, str(input_path), speed=speed)

            #output_path = OUTPUT_DIR / f"cloned_{ref_id}_{speaker_key_fmt}.wav"
            tone_color_converter.convert(
                audio_src_path=str(input_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(output_path)+speaker_key+".wav",
                message="Echoquest watermark"
            )

            results[speaker_key_fmt] = f"/download-cloned/{output_path.name+speaker_key}"+".wav"
            result_dict = {"data_uri": "/download-cloned/" + output_path.name+speaker_key+".wav"}

        except Exception as e:
            results[speaker_key_fmt] = f"Error: {str(e)}"

    return JSONResponse(content=result_dict)

@router.get("/download-cloned/{filename}")
async def download(filename: str):
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

def convert_to_wav(input_path: Path, output_path: Path):
    subprocess.run([
        "ffmpeg",
        "-y",  # Overwrite without asking
        "-i", str(input_path),
        "-ar", "44100",  # Sample rate
        "-ac", "1",      # Mono
        str(output_path)
    ], check=True)

