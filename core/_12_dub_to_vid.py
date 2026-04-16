import platform
import subprocess
import locale
import os

import cv2
import numpy as np
from rich.console import Console
from rich import print as rprint

from core._1_ytdlp import find_video_files
from core.asr_backend.audio_preprocess import normalize_audio_volume
from core.utils import *
from core.utils.models import *

console = Console()

DUB_VIDEO = "output/output_dub.mp4"
DUB_SUB_FILE = 'output/dub.srt'
DUB_AUDIO = 'output/dub.mp3'

TRANS_FONT_SIZE = 17
TRANS_FONT_NAME = 'Arial'
if platform.system() == 'Linux':
    TRANS_FONT_NAME = 'NotoSansCJK-Regular'
if platform.system() == 'Darwin':
    TRANS_FONT_NAME = 'Arial Unicode MS'

TRANS_FONT_COLOR = '&H00FFFF'
TRANS_OUTLINE_COLOR = '&H000000'
TRANS_OUTLINE_WIDTH = 1
TRANS_BACK_COLOR = '&H33000000'

SYS_ENC = locale.getpreferredencoding(False)


def run_ffmpeg(cmd, timeout=1800):
    """Run ffmpeg with visible diagnostics."""
    console.print(f"[cyan]🚀 Running FFmpeg:[/cyan] {' '.join(map(str, cmd))}")
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding=SYS_ENC,
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError("❌ FFmpeg 执行超时，可能卡住了")
    except Exception as e:
        raise RuntimeError(f"❌ FFmpeg 调用失败: {e}")

    if result.returncode != 0:
        if result.stdout:
            console.print("[yellow]FFmpeg stdout:[/yellow]")
            console.print(result.stdout[-4000:])
        if result.stderr:
            console.print("[red]FFmpeg stderr:[/red]")
            console.print(result.stderr[-8000:])
        raise RuntimeError(f"❌ FFmpeg 执行失败，退出码: {result.returncode}")

    return result


def _ensure_exists(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ {label}不存在: {path}")


def merge_video_audio():
    """Merge video + background audio + dubbed audio.
    If burn_subtitles is False, skip subtitle burn-in but still produce final video.
    """
    VIDEO_FILE = find_video_files()
    background_file = _BACKGROUND_AUDIO_FILE

    _ensure_exists(VIDEO_FILE, "视频文件")
    _ensure_exists(background_file, "背景音频文件")
    _ensure_exists(DUB_AUDIO, "配音文件")

    # Normalize dub audio
    normalized_dub_audio = 'output/normalized_dub.wav'
    normalize_audio_volume(DUB_AUDIO, normalized_dub_audio)
    _ensure_exists(normalized_dub_audio, "归一化配音文件")

    # Get target resolution
    video = cv2.VideoCapture(VIDEO_FILE)
    TARGET_WIDTH = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    TARGET_HEIGHT = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video.release()
    rprint(f"[bold green]Video resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}[/bold green]")

    burn_subtitles = load_key("burn_subtitles")

    base_video_filter = (
        f"[0:v]scale={TARGET_WIDTH}:{TARGET_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={TARGET_WIDTH}:{TARGET_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    )

    if burn_subtitles:
        _ensure_exists(DUB_SUB_FILE, "字幕文件")
        subtitle_filter = (
            f"subtitles={DUB_SUB_FILE}:force_style='FontSize={TRANS_FONT_SIZE},"
            f"FontName={TRANS_FONT_NAME},PrimaryColour={TRANS_FONT_COLOR},"
            f"OutlineColour={TRANS_OUTLINE_COLOR},OutlineWidth={TRANS_OUTLINE_WIDTH},"
            f"BackColour={TRANS_BACK_COLOR},Alignment=2,MarginV=27,BorderStyle=4'"
        )
        video_chain = f"{base_video_filter},{subtitle_filter}[v]"
        rprint("[bold green]Burning subtitles into video...[/bold green]")
    else:
        video_chain = f"{base_video_filter}[v]"
        rprint("[bold yellow]burn_subtitles=false，跳过字幕烧录，但继续合成视频与音频。[/bold yellow]")

    audio_chain = "[1:a][2:a]amix=inputs=2:duration=first:dropout_transition=3[a]"

    cmd = [
        'ffmpeg', '-y',
        '-i', VIDEO_FILE,
        '-i', background_file,
        '-i', normalized_dub_audio,
        '-filter_complex',
        f'{video_chain};{audio_chain}',
        '-map', '[v]',
        '-map', '[a]',
    ]

    if load_key("ffmpeg_gpu"):
        rprint("[bold green]Using GPU acceleration...[/bold green]")
        cmd.extend(['-c:v', 'h264_nvenc'])

    cmd.extend([
        '-c:a', 'aac',
        '-b:a', '96k',
        DUB_VIDEO
    ])

    run_ffmpeg(cmd)
    rprint(f"[bold green]Video and audio successfully merged into {DUB_VIDEO}[/bold green]")


if __name__ == '__main__':
    merge_video_audio()
