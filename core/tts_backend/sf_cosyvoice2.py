from openai import OpenAI
from pathlib import Path
from core.utils import *
from core.tts_backend.tts_reference_utils import ensure_reference_under_limit, wav_file_to_data_uri


@except_handler('Failed to generate audio using SiliconFlow TTS')
def cosyvoice_tts_for_videolingo(text, save_as, number, task_df):
    prompt_text = task_df.loc[task_df['number'] == number, 'origin'].values[0]
    API_KEY = load_key('sf_cosyvoice2.api_key')
    current_dir = Path.cwd()
    ref_audio_path = current_dir / f'output/audio/refers/{number}.wav'

    if not ref_audio_path.exists():
        ref_audio_path = current_dir / 'output/audio/refers/1.wav'
        if not ref_audio_path.exists():
            try:
                from core._9_refer_audio import extract_refer_audio_main
                print(f'参考音频文件不存在，尝试提取: {ref_audio_path}')
                extract_refer_audio_main()
            except Exception as e:
                print(f'提取参考音频失败: {str(e)}')
                raise

    safe_ref_audio_path, size_mb, duration_sec = ensure_reference_under_limit(ref_audio_path, max_mb=4.3, initial_max_sec=8.0)
    print(f'使用安全参考音频: {safe_ref_audio_path} | {duration_sec:.2f}s | {size_mb:.2f}MB')
    reference_data_uri = wav_file_to_data_uri(safe_ref_audio_path)

    client = OpenAI(api_key=API_KEY, base_url='https://api.siliconflow.cn/v1')
    save_path = Path(save_as)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model='FunAudioLLM/CosyVoice2-0.5B',
        voice='',
        input=text,
        response_format='wav',
        extra_body={'references': [{'audio': reference_data_uri, 'text': prompt_text}]}
    ) as response:
        response.stream_to_file(save_path)

    print(f'音频已成功保存至: {save_path}')
    return True
