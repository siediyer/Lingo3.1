import os
from pathlib import Path
from pydub import AudioSegment


def prepare_reference_audio(input_path, max_sec=8.0, sample_rate=16000, channels=1, prefix="ref_prepared"):
    """Prepare a reference audio for TTS APIs.

    - trims to max_sec
    - converts to mono, target sample rate, 16-bit PCM wav
    - exports to a sibling *_prepared.wav file
    Returns (prepared_path, size_mb, duration_sec)
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Reference audio not found: {input_path}")

    audio = AudioSegment.from_file(input_path)
    if max_sec and max_sec > 0:
        audio = audio[: int(max_sec * 1000)]

    audio = audio.set_frame_rate(sample_rate).set_channels(channels).set_sample_width(2)

    out_path = input_path.with_name(f"{input_path.stem}_{prefix}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(
        out_path,
        format="wav",
        parameters=["-acodec", "pcm_s16le", "-ar", str(sample_rate), "-ac", str(channels)],
    )

    size_mb = os.path.getsize(out_path) / 1024 / 1024
    duration_sec = len(audio) / 1000.0
    return str(out_path), size_mb, duration_sec


def ensure_reference_under_limit(input_path, max_mb=4.5, initial_max_sec=8.0, min_max_sec=2.0):
    """Iteratively shrink prepared reference audio until it's below max_mb."""
    max_sec = float(initial_max_sec)
    last = None
    while max_sec >= min_max_sec:
        prepared_path, size_mb, duration_sec = prepare_reference_audio(input_path, max_sec=max_sec)
        last = (prepared_path, size_mb, duration_sec)
        if size_mb <= max_mb:
            return prepared_path, size_mb, duration_sec
        max_sec -= 1.0
    return last


def wav_file_to_data_uri(wav_file_path):
    import base64
    with open(wav_file_path, 'rb') as audio_file:
        audio_content = audio_file.read()
    base64_audio = base64.b64encode(audio_content).decode('utf-8')
    return f"data:audio/wav;base64,{base64_audio}"
