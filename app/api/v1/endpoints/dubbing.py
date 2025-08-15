# Production-ready FastAPI for OpenVoice Voice Cloning

import os
import subprocess
import traceback

import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, APIRouter
from fastapi.responses import FileResponse, JSONResponse
from torchaudio._backend import ffmpeg

from app.utils.audio import save_upload_file

# Suppress SSL CA bundle warnings for torch hub
os.environ['CURL_CA_BUNDLE'] = ''

from melo.api import TTS
import nltk
from pydub import AudioSegment, silence


import sys
from pathlib import Path
from faster_whisper import WhisperModel

from transformers import MarianMTModel, MarianTokenizer

# Update this path to wherever the OpenVoice repo folder is located
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent  # This reaches echoquestcloning2/
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
OUTPUT_DIR = Path("dubbed")
CHECKPOINT_DIR = BASE_DIR /"checkpoints_v2"
BASE_SPEAKER_DIR = CHECKPOINT_DIR / "base_speakers" / "ses"
CONFIG_PATH = CHECKPOINT_DIR / "converter" / "config.json"
CKPT_PATH = CHECKPOINT_DIR / "converter" / "checkpoint.pth"

print(UPLOAD_DIR.parent)
UPLOAD_DIR.mkdir(exist_ok=True)
TEMP_UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

print(str(CONFIG_PATH))
# Setup model
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
tone_color_converter = ToneColorConverter(str(CONFIG_PATH), device=DEVICE)
tone_color_converter.load_ckpt(str(CKPT_PATH))

#ALLOWED_FORMATS = [".wav", ".mp3"]
ALLOWED_FORMATS = [".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".webm",".mov",".mp4",".avi",".mkv",".flv",".m3u8",".ts",".3gp",".wmv"]

# Supported language pairs: https://huggingface.co/Helsinki-NLP

LANG_MAP = {
    "EN": "en",
    "FR": "fr",
    "DE": "de",
    "ES": "es",
    "TA": "ta",
    "HI": "hi",
    "ZH": "zh",
    "JA": "jap",
    "KO": "ko"
}

MELOTTS_LANG_MAP = {
    "EN": "EN",
    "FR": "FR",
    "DE": "DE",
    "ES": "ES",
    "TA": "TA",
    "KO": "KR",
    "HI": "hi",
    "ZH": "ZH",
    "JA": "JP"
}
#
# Download entire model repo to a local directory
#model_dir = snapshot_download(
#    repo_id="guillaumekln/faster-whisper-small",
#    token='',
#    revision="main"  # Optional: Use a specific commit hash for reliability
#)

# Load model once (choose 'base', 'small', 'medium' as needed)
#model = WhisperModel("small", compute_type="float32")
model = WhisperModel("guillaumekln/faster-whisper-large-v2", compute_type="float32")  # Use float16 if your GPU supports it; else use "float32"


@router.post("/ai-dubbing")
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
    output_path = OUTPUT_DIR / f"dubbed_{Path(input_path).name}"
    try:
        if ext != ".wav":
            temp_input_path = save_upload_file(reference_audio, "temp_uploads",ext)
            convert_to_wav(Path(temp_input_path),UPLOAD_DIR / (Path(temp_input_path).name+".wav"))
            input_path = UPLOAD_DIR / (Path(temp_input_path).name+".wav")
        else:
            input_path = save_upload_file(reference_audio, "uploads",ext)

        trim_silence_with_pydub(Path(input_path))

        output_path = OUTPUT_DIR / f"dubbed_{Path(input_path).name}"

        target_se, _ = se_extractor.get_se(str(input_path), tone_color_converter, vad=False)
        target_se, audio_name = se_extractor.get_se(str(input_path), tone_color_converter, vad=True)
        #torch.save(source_se, "base_speakers/en-india.pth")

    except AssertionError as ae:
        # Catch specific assertion about audio being too short
        if "too short" in str(ae):
            raise HTTPException(status_code=400, detail="Input audio is too short or silent")
        else:
            raise HTTPException(status_code=400, detail=str(ae))
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Speaker embedding extraction failed: {str(e)}")

    language = custom_lang = language.upper()
    if language in ["EN-US","EN-BR","EN_INDIA","EN-AU"]:
        language = "EN"
    else:
        custom_lang = MELOTTS_LANG_MAP.get(language)


    if not input_text:
        # Step 1: Transcribe
        transcribed_text ,src_language= transcribe_audio(input_path)
        #if language.upper() == 'EN' and src_language.upper() == 'EN':
        if language.upper() == 'EN':
        # Step 2: Translate
            translated_text = transcribed_text
        else:
            #translated_text = translate_text(transcribed_text, language, src_language)
            translated_text = translate_text(transcribed_text, language)
    else:
        translated_text = input_text


    model = TTS(language=MELOTTS_LANG_MAP.get(language), device=DEVICE)
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
            #intermediate_path = OUTPUT_DIR / "intermediate_tts.wav"
            model.tts_to_file(translated_text, speaker_id, str(input_path), speed=speed)

            #output_path = OUTPUT_DIR / f"dubbed_{ref_id}_{speaker_key_fmt}.wav"
            tone_color_converter.convert(
                audio_src_path=str(input_path),
                src_se=source_se,
                tgt_se=target_se,
                output_path=str(output_path)+speaker_key+".wav",
                message="Echoquest watermark"
            )

            results[speaker_key_fmt] = f"/download-dubbed/{output_path.name+speaker_key}"+".wav"
            result_dict = {"data_uri": "/download-dubbed/" + output_path.name+speaker_key+".wav"}

        except Exception as e:
            traceback.print_exc()
            results[speaker_key_fmt] = f"Error: {str(e)}"

    return JSONResponse(content=result_dict)

@router.get("/download-dubbed/{filename}")
async def download(filename: str):
    file_path = OUTPUT_DIR / filename
    if file_path.exists():
        return FileResponse(file_path, media_type="audio/wav", filename=filename)
    raise HTTPException(status_code=404, detail="File not found")

def convert_to_wav(input_path: Path, output_path: Path):
    subprocess.run([
        "ffmpeg",
        "-y",                  # Overwrite output if exists
        "-i", str(input_path), # Input file
        "-ar", "16000",        # 16kHz sample rate
        "-ac", "1",            # Mono channel
        "-af", "loudnorm",     # Normalize loudness (optional but helps)
        str(output_path)
    ], check=True)

def trim_silence_with_pydub(wav_path: Path):
    sound = AudioSegment.from_wav(wav_path)

    # Detect non-silent chunks (min silence = 500ms, threshold = -40dBFS)
    chunks = silence.split_on_silence(
        sound,
        min_silence_len=500,
        silence_thresh=-40,
        keep_silence=200  # optional padding around spoken parts
    )

    if not chunks:
        print("No non-silent chunks found.")
        return

    # Recombine the non-silent chunks
    processed_audio = AudioSegment.silent(duration=300)
    for chunk in chunks:
        processed_audio += chunk + AudioSegment.silent(duration=150)

    # Overwrite the input file with trimmed version
    processed_audio.export(wav_path, format="wav")



def transcribe_audio(audio_path: str) -> str:
    segments, info = model.transcribe(str(audio_path), beam_size=5, task="translate")#Only supports english translation
    full_text = " ".join([segment.text for segment in segments])
    return full_text.strip(), info.language

def load_translation_model(src: str, tgt: str):
    if tgt == 'ko':
        src = "tc-big-"+src
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def translate_text(text: str, target_lang: str, source_lang: str = "en") -> str:
    src = LANG_MAP.get(source_lang.upper(), "en")
    tgt = LANG_MAP.get(target_lang.upper(), "en")
    tokenizer, model = load_translation_model(src, tgt)

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
    return translated_text.strip()

def enhance_output(path):
    enhanced_path = str(path).replace(".wav", "_enhanced.wav")
    (
        ffmpeg
        .input(str(path))
        .output(enhanced_path, af="loudnorm, dynaudnorm, highpass=f=200, lowpass=f=3000")
        .run(overwrite_output=True)
    )
    return enhanced_path
