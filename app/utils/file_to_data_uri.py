import base64
from pathlib import Path


def file_to_data_uri(file_path: Path) -> str:
    """
    Converts a file to a base64-encoded data URI.

    Args:
        file_path (Path): Path to the audio file (.wav)

    Returns:
        str: Data URI with base64-encoded content.
    """
    mime_type = "audio/wav"  # Change if you're using a different format
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def encode_to_data_uri(file_path, mime_type="audio/wav"):
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    return f"data:{mime_type};base64,{encoded}"
