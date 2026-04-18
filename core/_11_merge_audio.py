
import os
import ast
import re
import math
import locale
import shutil
import tempfile
import subprocess
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import print as rprint

from core.utils import *
from core.utils.models import *

console = Console()

DUB_VOCAL_FILE = 'output/dub.mp3'
DUB_SUB_FILE = 'output/dub.srt'
OUTPUT_FILE_TEMPLATE = f"{_AUDIO_SEGS_DIR}/{{}}.wav"

SYS_ENC = locale.getpreferredencoding(False)
TARGET_SR = 16000
TARGET_CH = 1


def safe_parse(value):
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return value

    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    cleaned = text
    wrapper_patterns = [
        r'np\.float64\(([^()]*)\)',
        r'float64\(([^()]*)\)',
        r'np\.float32\(([^()]*)\)',
        r'float32\(([^()]*)\)',
        r'np\.int64\(([^()]*)\)',
        r'int64\(([^()]*)\)',
        r'np\.int32\(([^()]*)\)',
        r'int32\(([^()]*)\)',
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

    try:
        return ast.literal_eval(cleaned)
    except Exception:
        return value


def ensure_parent_dir(file_path):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def run_ffmpeg(cmd, timeout=1800):
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
        raise RuntimeError(f"❌ FFmpeg 执行超时：{' '.join(map(str, cmd))}")
    except Exception as e:
        raise RuntimeError(f"❌ FFmpeg 调用失败：{e}")

    if result.returncode != 0:
        if result.stdout:
            console.print("[yellow]FFmpeg stdout:[/yellow]")
            console.print(result.stdout[-4000:])
        if result.stderr:
            console.print("[red]FFmpeg stderr:[/red]")
            console.print(result.stderr[-8000:])
        raise RuntimeError(f"❌ FFmpeg 执行失败，退出码: {result.returncode}")

    return result


def load_and_flatten_data(excel_file):
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
    audios = []
    for row_idx, (_, row) in enumerate(df.iterrows(), 1):
        number = row['number']
        line_items = safe_parse(row['lines'])
        if not isinstance(line_items, (list, tuple)):
            raise RuntimeError(f"❌ 第 {row_idx} 行 lines 字段不是列表: {repr(row['lines'])}")

        for line_index in range(len(line_items)):
            audios.append(OUTPUT_FILE_TEMPLATE.format(f"{number}_{line_index}"))
    return audios


def format_time(t):
    return (
        f"{int(t//3600):02d}:{int((t%3600)//60):02d}:{int(t%60):02d},"
        f"{int((t*1000)%1000):03d}"
    )


def create_srt_subtitle():
    df, lines, new_sub_times = load_and_flatten_data(_8_1_AUDIO_TASK)
    ensure_parent_dir(DUB_SUB_FILE)

    with open(DUB_SUB_FILE, 'w', encoding='utf-8') as f:
        for i, ((start, end), line) in enumerate(zip(new_sub_times, lines), 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{line}\n\n")

    rprint(f"[bold green]✅ Subtitle file created: {DUB_SUB_FILE}[/bold green]")


def build_concat_plan(audios, new_sub_times):
    """
    构建拼接计划：
    [('silence', 1.23), ('audio', '...wav'), ...]
    """
    plan = []
    prev_end = 0.0

    for idx, (audio_file, time_range) in enumerate(zip(audios, new_sub_times), 1):
        start_time, end_time = time_range

        if not os.path.exists(audio_file):
            console.print(f"[yellow]⚠️ 第 {idx} 段不存在，跳过: {audio_file}[/yellow]")
            continue
        if os.path.getsize(audio_file) == 0:
            console.print(f"[yellow]⚠️ 第 {idx} 段为空文件，跳过: {audio_file}[/yellow]")
            continue

        gap = max(0.0, float(start_time) - float(prev_end))
        if gap > 0.001:
            plan.append(("silence", gap))

        plan.append(("audio", audio_file))
        prev_end = float(end_time)

    return plan


def create_silence_wav(duration_sec, out_path):
    duration_sec = max(0.001, float(duration_sec))
    ensure_parent_dir(out_path)

    cmd = [
        'ffmpeg', '-y',
        '-f', 'lavfi',
        '-i', f'anullsrc=r={TARGET_SR}:cl=mono',
        '-t', f'{duration_sec:.3f}',
        '-ar', str(TARGET_SR),
        '-ac', str(TARGET_CH),
        '-c:a', 'pcm_s16le',
        out_path
    ]
    run_ffmpeg(cmd, timeout=300)


def ensure_uniform_wav(src_path, out_path):
    """
    为 concat demuxer 统一格式：
    PCM s16le / 16k / mono / wav
    """
    ensure_parent_dir(out_path)
    cmd = [
        'ffmpeg', '-y',
        '-i', src_path,
        '-ar', str(TARGET_SR),
        '-ac', str(TARGET_CH),
        '-c:a', 'pcm_s16le',
        out_path
    ]
    run_ffmpeg(cmd, timeout=300)


def generate_concat_inputs(plan, work_dir):
    """
    生成 concat 所需的标准化 wav 与列表文件。
    """
    concat_list_path = os.path.join(work_dir, 'concat_list.txt')
    inputs_dir = os.path.join(work_dir, 'inputs')
    os.makedirs(inputs_dir, exist_ok=True)

    silence_cache = {}
    lines = []
    seq = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn()
    ) as progress:
        task = progress.add_task("🎵 Preparing concat inputs...", total=len(plan))

        for kind, value in plan:
            seq += 1
            if kind == "silence":
                duration_ms = int(round(float(value) * 1000))
                if duration_ms not in silence_cache:
                    silence_path = os.path.join(inputs_dir, f"silence_{duration_ms}ms.wav")
                    create_silence_wav(duration_ms / 1000.0, silence_path)
                    silence_cache[duration_ms] = silence_path
                use_path = silence_cache[duration_ms]
            else:
                src_audio = value
                use_path = os.path.join(inputs_dir, f"{seq:06d}.wav")
                ensure_uniform_wav(src_audio, use_path)

            safe_path = use_path.replace("\\", "/").replace("'", r"'\''")
            lines.append(f"file '{safe_path}'\n")
            progress.advance(task)

    with open(concat_list_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

    return concat_list_path


def concat_to_mp3(concat_list_path, output_mp3):
    ensure_parent_dir(output_mp3)

    temp_wav = os.path.join(os.path.dirname(output_mp3) or '.', 'dub_concat_temp.wav')

    # 先无损拼接为 wav
    cmd_concat = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_list_path,
        '-c', 'copy',
        temp_wav
    ]
    run_ffmpeg(cmd_concat, timeout=1800)

    # 再统一导出 mp3
    cmd_mp3 = [
        'ffmpeg', '-y',
        '-i', temp_wav,
        '-ar', str(TARGET_SR),
        '-ac', str(TARGET_CH),
        '-b:a', '64k',
        output_mp3
    ]
    run_ffmpeg(cmd_mp3, timeout=1800)

    if os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
        except Exception:
            pass


def merge_full_audio():
    console.print("\n[bold cyan]🎬 Starting EXTREME audio merging process (FFmpeg concat)...[/bold cyan]")

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

    plan = build_concat_plan(audios, new_sub_times)
    if not plan:
        raise RuntimeError("❌ 没有可用的音频拼接计划")

    console.print(f"[bold green]✅ Concat plan entries: {len(plan)}[/bold green]")

    with tempfile.TemporaryDirectory(prefix="videolingo_concat_") as work_dir:
        concat_list_path = generate_concat_inputs(plan, work_dir)
        concat_to_mp3(concat_list_path, DUB_VOCAL_FILE)

    console.print(f"[bold green]✅ Audio file successfully merged![/bold green]")
    console.print(f"[bold green]📁 Output file: {DUB_VOCAL_FILE}[/bold green]")


if __name__ == "__main__":
    merge_full_audio()
