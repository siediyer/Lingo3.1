import os
import warnings
import time
import subprocess
import torch
import functools

warnings.filterwarnings("ignore")

# =============================================================================
# Compatibility shim — applied BEFORE importing whisperx
# =============================================================================

_original_torch_load = torch.load

@functools.wraps(_original_torch_load)
def _patched_torch_load(*args, **kwargs):
    if kwargs.get("weights_only") is None:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# =============================================================================
# Now safe to import whisperx and the rest of the application
# =============================================================================
import whisperx
from whisperx.audio import load_audio as _whisperx_load_audio, SAMPLE_RATE as _WHISPERX_SR
from rich import print as rprint
from core.utils import *

MODEL_DIR = load_key("model_dir")


@except_handler("failed to check hf mirror", default_return=None)
def check_hf_mirror():
    mirrors = {'Official': 'huggingface.co', 'Mirror': 'hf-mirror.com'}
    fastest_url = f"https://{mirrors['Official']}"
    best_time = float('inf')
    rprint("[cyan]🔍 Checking HuggingFace mirrors...[/cyan]")

    for name, domain in mirrors.items():
        if os.name == 'nt':
            cmd = ['ping', '-n', '1', '-w', '3000', domain]
        else:
            cmd = ['ping', '-c', '1', '-W', '3', domain]

        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="gb18030",
            errors="replace"
        )
        response_time = time.time() - start

        if result.returncode == 0:
            if response_time < best_time:
                best_time = response_time
                fastest_url = f"https://{domain}"
            rprint(f"[green]✓ {name}:[/green] {response_time:.2f}s")

    if best_time == float('inf'):
        rprint("[yellow]⚠️ All mirrors failed, using default[/yellow]")

    rprint(f"[cyan]🚀 Selected mirror:[/cyan] {fastest_url} ({best_time:.2f}s)")
    return fastest_url


@except_handler("WhisperX processing error:")
def transcribe_audio(raw_audio_file, vocal_audio_file, start, end):
    os.environ['HF_ENDPOINT'] = check_hf_mirror()
    WHISPER_LANGUAGE = load_key("whisper.language")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rprint(f"🚀 Starting WhisperX using device: {device} ...")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        if hasattr(torch.backends.cudnn, "allow_tf32"):
            torch.backends.cudnn.allow_tf32 = True

        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        # 🚀 暴力模式
        batch_size = 48
        compute_type = "float16"

        rprint(
            f"[cyan]🎮 GPU memory:[/cyan] {gpu_mem:.2f} GB, "
            f"[cyan]📦 Batch size:[/cyan] {batch_size}, "
            f"[cyan]⚙️ Compute type:[/cyan] {compute_type}"
        )
    else:
        batch_size = 1
        compute_type = "int8"
        rprint(
            f"[cyan]📦 Batch size:[/cyan] {batch_size}, "
            f"[cyan]⚙️ Compute type:[/cyan] {compute_type}"
        )

    rprint(f"[green]▶️ Starting WhisperX for segment {start:.2f}s to {end:.2f}s...[/green]")

    if WHISPER_LANGUAGE == 'zh':
        model_name = "Huan69/Belle-whisper-large-v3-zh-punct-fasterwhisper"
        local_model = os.path.join(MODEL_DIR, "Belle-whisper-large-v3-zh-punct-fasterwhisper")
    else:
        model_name = load_key("whisper.model")
        local_model = os.path.join(MODEL_DIR, model_name)

    if os.path.exists(local_model):
        rprint(f"[green]📥 Loading local WHISPER model:[/green] {local_model} ...")
        model_name = local_model
    else:
        rprint(f"[green]📥 Using WHISPER model from HuggingFace:[/green] {model_name} ...")

    vad_options = {"vad_onset": 0.500, "vad_offset": 0.363}
    asr_options = {"temperatures": [0], "initial_prompt": ""}
    whisper_language = None if 'auto' in WHISPER_LANGUAGE else WHISPER_LANGUAGE

    rprint("[bold yellow] You can ignore warning of model version mismatch[/bold yellow]")

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=whisper_language,
        vad_options=vad_options,
        asr_options=asr_options,
        download_root=MODEL_DIR
    )

    def load_audio_segment(audio_file, start, end):
        full_audio = _whisperx_load_audio(audio_file, sr=_WHISPERX_SR)
        start_sample = int(start * _WHISPERX_SR)
        end_sample = int(end * _WHISPERX_SR)
        return full_audio[start_sample:end_sample]

    raw_audio_segment = load_audio_segment(raw_audio_file, start, end)
    vocal_audio_segment = load_audio_segment(vocal_audio_file, start, end)

    transcribe_start_time = time.time()
    rprint("[bold green]Processing...[/bold green]")
    result = model.transcribe(raw_audio_segment, batch_size=batch_size, print_progress=True)
    rprint(f"[cyan]⏱️ time transcribe:[/cyan] {time.time() - transcribe_start_time:.2f}s")

    del model
    torch.cuda.empty_cache()

    update_key("whisper.language", result['language'])

    align_start_time = time.time()
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, vocal_audio_segment, device, return_char_alignments=False)
    rprint(f"[cyan]⏱️ time align:[/cyan] {time.time() - align_start_time:.2f}s")

    torch.cuda.empty_cache()
    del model_a

    for segment in result['segments']:
        segment['start'] += start
        segment['end'] += start

    return result
