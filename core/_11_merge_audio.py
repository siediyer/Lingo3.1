import os
import ast
import re
import locale
import pandas as pd
import subprocess
from pydub import AudioSegment
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.console import Console
from rich import print as rprint

from core.utils import *
from core.utils.models import *

console = Console()

DUB_VOCAL_FILE = 'output/dub.mp3'
DUB_SUB_FILE = 'output/dub.srt'
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"

SYS_ENC = locale.getpreferredencoding(False)


def safe_parse(value):
    """Robust parser for Excel cell data, including nested np.float64(...) wrappers."""
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    # 1) Try direct literal parsing first
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    cleaned = text

    # 2) Replace common wrappers with just their inner content
    wrapper_patterns = [
        r'np\.float64\(([^()]*)\)',
        r'float64\(([^()]*)\)',
        r'np\.float32\(([^()]*)\)',
        r'float32\(([^()]*)\)',
        r'np\.int64\(([^()]*)\)',
        r'int64\(([^()]*)\)',
        r'np\.int32\(([^()]*)\)',
        r'int32\(([^()]*)\)',
        r'np\.int16\(([^()]*)\)',
        r'int16\(([^()]*)\)',
        r'np\.int8\(([^()]*)\)',
        r'int8\(([^()]*)\)',
        r'np\.bool_\(([^()]*)\)',
        r'bool_\(([^()]*)\)',
        r'Timestamp\(([^()]*)\)',
    ]

    changed = True
    while changed:
        changed = False
        for pattern in wrapper_patterns:
            new_cleaned = re.sub(pattern, r'\1', cleaned)
            if new_cleaned != cleaned:
                cleaned = new_cleaned
                changed = True

    # 3) Retry parsing after wrapper removal
    try:
        return ast.literal_eval(cleaned)
    except Exception:
        pass

    # 4) Fallback for quoted scalar values left by Timestamp('...')
    if (cleaned.startswith("'") and cleaned.endswith("'")) or (cleaned.startswith('"') and cleaned.endswith('"')):
        try:
            return ast.literal_eval(cleaned)
        except Exception:
            return cleaned[1:-1]

    return value


def load_and_flatten_data(excel_file):
    """Load and flatten Excel data."""
    df = pd.read_excel(excel_file)

    lines = []
    for idx, line in enumerate(df['lines'].tolist(), 1):
        parsed = safe_parse(line)
        if not isinstance(parsed, (list, tuple)):
            raise RuntimeError(f"❌ 第 {idx} 行 lines 解析失败: {repr(line)}")
        lines.extend(parsed)

    new_sub_times = []
    for idx, item in enumerate(df['new_sub_times'].tolist(), 1):
        parsed = safe_parse(item)
        if not isinstance(parsed, (list, tuple)):
            raise RuntimeError(f"❌ 第 {idx} 行 new_sub_times 解析失败: {repr(item)}")
        new_sub_times.extend(parsed)

    return df, lines, new_sub_times


def get_audio_files(df):
    """Generate a list of audio file paths."""
    audios = []
    for row_idx, (_, row) in enumerate(df.iterrows(), 1):
        number = row['number']
        line_items = safe_parse(row['lines'])

        if not isinstance(line_items, (list, tuple)):
            raise RuntimeError(f"❌ 第 {row_idx} 行 lines 字段不是列表: {repr(row['lines'])}")

        line_count = len(line_items)
        for line_index in range(line_count):
            temp_file = OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}")
            audios.append(temp_file)
    return audios


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_ffmpeg(cmd, timeout=300):
    """Run ffmpeg safely and print useful diagnostics."""
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
            check=False
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"❌ FFmpeg 执行超时，可能卡住了：{' '.join(map(str, cmd))}")
    except Exception as e:
        raise RuntimeError(f"❌ FFmpeg 调用失败：{e}")

    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip()
        stdout_text = (result.stdout or "").strip()

        if stdout_text:
            console.print("[yellow]FFmpeg stdout:[/yellow]")
            console.print(stdout_text[-4000:])

        if stderr_text:
            console.print("[red]FFmpeg stderr:[/red]")
            console.print(stderr_text[-8000:])

        raise RuntimeError(f"❌ FFmpeg 执行失败，退出码: {result.returncode}")

    return result


def process_audio_segment(audio_file):
    """Process a single audio segment with MP3 compression."""
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"音频文件不存在: {audio_file}")

    if os.path.getsize(audio_file) == 0:
        raise RuntimeError(f"音频文件为空: {audio_file}")

    temp_file = f"{audio_file}_temp.mp3"
    ensure_parent_dir(temp_file)

    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-i', audio_file,
        '-ar', '16000',
        '-ac', '1',
        '-b:a', '64k',
        temp_file
    ]

    console.print(f"[blue]🎧 Processing segment:[/blue] {audio_file}")
    run_ffmpeg(ffmpeg_cmd, timeout=300)

    if not os.path.exists(temp_file):
        raise RuntimeError(f"❌ FFmpeg 未生成临时文件: {temp_file}")

    if os.path.getsize(temp_file) == 0:
        raise RuntimeError(f"❌ FFmpeg 生成了空临时文件: {temp_file}")

    try:
        audio_segment = AudioSegment.from_mp3(temp_file)
    except Exception as e:
        raise RuntimeError(f"❌ 读取临时 MP3 失败: {temp_file}, 错误: {e}")
    finally:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass

    return audio_segment


def merge_audio_segments(audios, new_sub_times, sample_rate):
    """Merge all audio segments into one track."""
    merged_audio = AudioSegment.silent(duration=0, frame_rate=sample_rate)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn()
    ) as progress:
        merge_task = progress.add_task("🎵 Merging audio segments...", total=len(audios))

        for i, (audio_file, time_range) in enumerate(zip(audios, new_sub_times)):
            console.print(f"[blue]➡️ Merging {i+1}/{len(audios)}:[/blue] {audio_file}")

            if not os.path.exists(audio_file):
                console.print(f"[bold yellow]⚠️ Warning: File {audio_file} does not exist, skipping...[/bold yellow]")
                progress.advance(merge_task)
                continue

            if os.path.getsize(audio_file) == 0:
                console.print(f"[bold yellow]⚠️ Warning: File {audio_file} is empty, skipping...[/bold yellow]")
                progress.advance(merge_task)
                continue

            audio_segment = process_audio_segment(audio_file)
            start_time, end_time = time_range

            if i > 0:
                prev_end = new_sub_times[i - 1][1]
                silence_duration = start_time - prev_end
                if silence_duration > 0:
                    silence = AudioSegment.silent(
                        duration=int(silence_duration * 1000),
                        frame_rate=sample_rate
                    )
                    merged_audio += silence
            elif start_time > 0:
                silence = AudioSegment.silent(
                    duration=int(start_time * 1000),
                    frame_rate=sample_rate
                )
                merged_audio += silence

            merged_audio += audio_segment
            progress.advance(merge_task)

    return merged_audio


def create_srt_subtitle():
    """Create subtitle file."""
    df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    ensure_parent_dir(DUB_SUB_FILE)

    with open(DUB_SUB_FILE, 'w', encoding='utf-8') as f:
        for i, ((start_time, end_time), line) in enumerate(zip(new_sub_times, lines), 1):
            start_str = (
                f"{int(start_time // 3600):02d}:"
                f"{int((start_time % 3600) // 60):02d}:"
                f"{int(start_time % 60):02d},"
                f"{int((start_time * 1000) % 1000):03d}"
            )
            end_str = (
                f"{int(end_time // 3600):02d}:"
                f"{int((end_time % 3600) // 60):02d}:"
                f"{int(end_time % 60):02d},"
                f"{int((end_time * 1000) % 1000):03d}"
            )

            f.write(f"{i}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{line}\n\n")

    rprint(f"[bold green]✅ Subtitle file created: {DUB_SUB_FILE}[/bold green]")


def merge_full_audio():
    """Main function: process the complete audio merging process."""
    console.print("\n[bold cyan]🎬 Starting audio merging process...[/bold cyan]")

    with console.status("[bold cyan]📊 Loading data from Excel...[/bold cyan]"):
        df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    console.print("[bold green]✅ Data loaded successfully[/bold green]")

    with console.status("[bold cyan]🔍 Getting audio file list...[/bold cyan]"):
        audios = get_audio_files(df)
    console.print(f"[bold green]✅ Found {len(audios)} audio segments[/bold green]")

    if not audios:
        console.print("[bold red]❌ Error: No audio segments found![/bold red]")
        return

    with console.status("[bold cyan]📝 Generating subtitle file...[/bold cyan]"):
        create_srt_subtitle()

    if not os.path.exists(audios[0]):
        console.print(f"[bold red]❌ Error: First audio file {audios[0]} does not exist![/bold red]")
        return

    sample_rate = 16000
    console.print(f"[bold green]✅ Sample rate: {sample_rate}Hz[/bold green]")

    console.print("[bold cyan]🔄 Starting audio merge process...[/bold cyan]")
    merged_audio = merge_audio_segments(audios, new_sub_times, sample_rate)

    ensure_parent_dir(DUB_VOCAL_FILE)
    with console.status("[bold cyan]💾 Exporting final audio file...[/bold cyan]"):
        merged_audio = merged_audio.set_frame_rate(16000).set_channels(1)
        merged_audio.export(DUB_VOCAL_FILE, format="mp3", parameters=["-b:a", "64k"])

    console.print(f"[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output file: {DUB_VOCAL_FILE}[/bold green]")


if __name__ == "__main__":
    merge_full_audio()
