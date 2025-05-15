# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import os
import gc
import shutil
import glob
import json
import re
import tempfile
from collections import OrderedDict
from importlib.resources import files

import click
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from cached_path import cached_path
from transformers import AutoModelForCausalLM, AutoTokenizer

import subprocess
from unidecode import unidecode
import unicodedata
import tempfile
from pydub import AudioSegment, silence

# Set up GPU utilization
if torch.cuda.is_available():
    gpu_ids = list(range(torch.cuda.device_count()))
else:
    gpu_ids = []

try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, DiT_v0, UNetT
from f5_tts.infer.utils_infer import (
    transcribe,
    load_vocoder,
    load_model,
    infer_process,
    remove_silence_for_generated_wav,
    save_spectrogram,
)


DEFAULT_TTS_MODEL = "F5-TTS_v1"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


# load models

vocoder = load_vocoder()


def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts():
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4, text_mask_padding=False, pe_attn_head=1)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


def load_custom(model_ver: str, ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    if model_ver == "DiT_v0":
        return load_model(DiT_v0, model_cfg, ckpt_path, vocab_file=vocab_path)
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None

# Load voice configurations from JSON file
voices = {}
try:
    with open('voice_configs.json', 'r', encoding='utf-8') as f:
        voices = json.load(f)
except FileNotFoundError:
    print("Warning: voice_configs.json not found. Using default configuration.")

def process_text(text):
    if not text:
        gr.Warning("Vui lòng điền text để xử lý!")
        return text

    # Unicode and lowercase
    text = unicodedata.normalize('NFC', text)
    text = text.lower()

    # Handle ellipsis patterns in one pass
    text = re.sub(r'(\.\s*){3,}', '…', text)
    text = re.sub(r'(\.\s*){2}', '.', text)

    # Remove all quotations, parantheses
    text = re.sub(r'[\(\)\"\'\“\”\<\>\`]', '', text)

    # Replace colons and semicolons with periods
    text = re.sub(r'[\:\;]', '.', text)
    
    # Remove bullet points
    text = re.sub(r"^-\s*", "", text, flags=re.MULTILINE)

    # Capitalize the first letter after a period
    text = re.sub(r'[\.?!]\s*([^\W\d_])', lambda match: f'. {match.group(1).upper()}', text, flags=re.UNICODE)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    def capitalize(text):
        match = re.match(r'([^\w]*\s*)(\w)(.*)', text)
        if match:
            return match.group(1) + match.group(2).upper() + match.group(3)
        return text

    # Apply capitalization with regex (faster than looping)
    text = capitalize(text)

    word_pattern = r'[\w\d\-\/\\\$\%\&]+'
    punct_pattern = r'[,:;.…]|[!?]+'

    # Combine spacing adjustments into fewer operations if possible
    spacing_pattern = re.compile(f'({word_pattern})(?!\\s)({punct_pattern})|({punct_pattern})(?!\\s)({word_pattern})|({punct_pattern})(?!\\s)({punct_pattern})')

    def spacing_replacement(match):
        # Group 1 and 2: word followed by punctuation
        if match.group(1) and match.group(2):
            return f"{match.group(1)} {match.group(2)}"
        # Group 3 and 4: punctuation followed by word
        elif match.group(3) and match.group(4):
            return f"{match.group(3)} {match.group(4)}"
        # Group 5 and 6: punctuation followed by punctuation
        elif match.group(5) and match.group(6):
            return f"{match.group(5)} {match.group(6)}"
        return match.group(0)

    while True:
        new_text = spacing_pattern.sub(spacing_replacement, text)
        if new_text == text:  # If no changes were made, stop
            break
        text = new_text

    # Clean up the ending
    text = text.rstrip()
    if text.endswith((',', ';', ':')):
        text = text[:-1] + '.'
    elif not text.endswith(('.', '!', '?', '…')):
        text += ' .'

    return text

@gpu_decorator
def generate_response(messages, model, tokenizer):
    """Generate response using Qwen"""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
    )

    generated_ids = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


@gpu_decorator
def infer(
    ref_audio,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=gr.Info,
):
    if not gen_text.strip():
        gr.Warning("Vui lòng điền text để đọc!")
        return gr.update(), gr.update(), ref_text
    
    gen_text = process_text(gen_text)

    if model == DEFAULT_TTS_MODEL:
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        global E2TTS_ema_model
        if E2TTS_ema_model is None:
            show_info("Đang tải E2-TTS model...")
            E2TTS_ema_model = load_e2tts()
        ema_model = E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[2]:
            show_info("Đang tải Custom TTS model...")
            custom_ema_model = load_custom(model[1], model[2], vocab_path=model[3], model_cfg=model[4])
            pre_custom_path = model[2]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
        progress=gr.Progress(),
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Save the spectrogram
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (final_sample_rate, final_wave), spectrogram_path

with gr.Blocks() as app_new_voice:
    gr.Markdown("# Tạo giọng của bạn")
    
    with gr.Row():
        voice_name = gr.Textbox(label="Tên giọng", placeholder="Create new voice ...")
        voice_lang = gr.Radio(
            choices=["vi", "en", "zh"], label="Ngôn ngữ",
            value="vi"
        )
        with gr.Column():
            create_name_btn = gr.Button("Bước 1: Tạo tên giọng", variant="secondary", size="lg")
            delete_name_btn = gr.Button("Xóa giọng", variant="stop", size="lg")

    gr.Markdown("# Xử lý audio và text")
    
    with gr.Row():
        ref_audio = gr.Audio(label="Reference Audio", type="filepath", show_download_button=True)
        ref_text = gr.Textbox(label="Reference Text")
    
    with gr.Row():
        pp_audio_btn = gr.Button("Bước 2: Reference Audio", variant="secondary")
        pp_ref_text = gr.Button("Bước 3: Reference Text", variant="secondary")
    
    create_voice_btn = gr.Button("Bước 4: Tạo giọng", variant="primary")
    
    voice_path = gr.State()
    
    def remove_silence_edges(audio, silence_threshold=-42):
        # Remove silence from the start
        non_silent_start_idx = silence.detect_leading_silence(audio, silence_threshold=silence_threshold)
        audio = audio[non_silent_start_idx:]

        # Remove silence from the end
        non_silent_end_duration = audio.duration_seconds
        for ms in reversed(audio):
            if ms.dBFS > silence_threshold:
                break
            non_silent_end_duration -= 0.001
        trimmed_audio = audio[: int(non_silent_end_duration * 1000)]

        return trimmed_audio

    def process_melband(audio_files, output_dir, infer_path, config_path, model_path):
        command = [
            "python",
            infer_path,
            "--config_path", config_path,
            "--model_path", model_path,
            "--store_dir", output_dir,
            "--audio_files"
        ] + audio_files

        # Add GPU IDs from the argument parser if provided
        if gpu_ids:  # gpu_ids will be a list from the argument parser
            command.extend(["--device_ids"] + list(map(str, gpu_ids)))  # Convert integers to strings

        # Run the inference
        try:
            print(f"Running command: {' '.join(command)}")
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing with Mel-Band-Roformer: {str(e)}")

    def process_voice_name(voice_name):
        voice_name = unidecode(voice_name.lower()).replace(" ", "_")
        voice_store = "voices"
        voice_dir = os.path.join(voice_store, voice_name)
        if os.path.exists(voice_dir):
            gr.Warning(f"Kiểu giọng đã tồn tại, vui lòng chọn kiểu giọng khác!")
            return None
        os.makedirs(voice_dir)
        voice_path = os.path.join(voice_dir, f"{voice_name}.wav")
        gr.Info("Tạo tên giọng mới hoàn tất")
        return voice_path
    
    def delete_voice(voice_name, voice_lang):
        voice_name = unidecode(voice_name.lower()).replace(" ", "_")
        voice_store = "voices"
        voice_dir = os.path.join(voice_store, voice_name)

        if os.path.exists(voice_dir):
            shutil.rmtree(voice_dir)

        voice_configs = "voice_configs.json"
        try:
            with open(voice_configs, 'r', encoding='utf-8') as f:
                voices = json.load(f)
        except FileNotFoundError:
            voices = {}

        voice_key = f"{voice_lang}_{voice_name}"

        breakpoint()
        
        if voice_key not in voices:
            gr.Warning("Kiểu giọng không tồn tại!")
            return gr.update(), gr.update()
        
        del voices[voice_key]
        with open(voice_configs, 'w', encoding='utf-8') as f:
            json.dump(voices, f, indent=2, ensure_ascii=False)

        gr.Info("Xóa giọng thành công")
        return "", "vi"

    def process_ref_audio(ref_audio, voice_path, clip_short=True, show_info=gr.Info, melband_repo_path=None, deecho_model_path=None, min_duration=5000, max_duration=10000, start_buffer=300, end_silence=700, sample_rate=24000, is_mono=True):
        if not ref_audio:
            gr.Warning("Vui lòng thêm reference audio!")
            return gr.update(), gr.update(), gr.update()
        
        show_info("Chuẩn hóa audio ...")
        audio_length_limit = max_duration // 1000
        with tempfile.NamedTemporaryFile(delete=True, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio)

            if clip_short:
                # 1. try to find long silence for clipping
                non_silent_segs = silence.split_on_silence(
                    aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
                )
                non_silent_wave = AudioSegment.silent(duration=0)
                for non_silent_seg in non_silent_segs:
                    if len(non_silent_wave) > min_duration and len(non_silent_wave + non_silent_seg) > max_duration:
                        show_info(f"Audio quá {audio_length_limit}s, cắt ngắn. (1)")
                        break
                    non_silent_wave += non_silent_seg

                # 2. try to find short silence for clipping if 1. failed
                if len(non_silent_wave) > max_duration:
                    non_silent_segs = silence.split_on_silence(
                        aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
                    )
                    non_silent_wave = AudioSegment.silent(duration=0)
                    for non_silent_seg in non_silent_segs:
                        if len(non_silent_wave) > min_duration and len(non_silent_wave + non_silent_seg) > max_duration:
                            show_info(f"Audio quá {audio_length_limit}s, cắt ngắn. (2)")
                            break
                        non_silent_wave += non_silent_seg

                aseg = non_silent_wave

                # 3. if no proper silence found for clipping
                if len(aseg) > max_duration:
                    aseg = aseg[:max_duration]
                    show_info(f"Audio quá {audio_length_limit}s, cắt ngắn. (3)")
                
            aseg = AudioSegment.silent(duration=start_buffer) + remove_silence_edges(aseg) + AudioSegment.silent(duration=end_silence)
            aseg = aseg.set_frame_rate(sample_rate)
            aseg = aseg.set_channels(1 if is_mono else 2)
            aseg.export(voice_path, format="wav")
            processed_ref_audio = voice_path

        # Set up Mel-Band-Roformer paths
        if melband_repo_path is None:
            melband_repo_path = files("f5_tts")._paths[0].resolve().parents[1] / "Mel-Band-Roformer-Vocal-Model"
        melband_infer_path = os.path.join(melband_repo_path, 'inference.py')
        melband_config_path = os.path.join(melband_repo_path, 'configs', 'config_vocals_mel_band_roformer.yaml')
        melband_model_path = os.path.join(melband_repo_path, 'MelBandRoformer.ckpt')

        # # Set up DeEcho path
        # if deecho_model_path is None:
        #    deecho_model_path='UVR-De-Echo-Normal.pth'

        # Step 2: Vocal extraction
        show_info("Trích xuất vocals ...")
        process_melband([processed_ref_audio], os.path.dirname(voice_path), melband_infer_path, melband_config_path, melband_model_path)
        
        # # Step 3: De-echo processing
        # show_info("Remove echo if any ...")
        # deecho_output_files = process_deecho(audio_files, tmp_dir, deecho_model_path, sample_rate, gpu_ids)

        gr.Warning("Reference audio vừa được sửa lại, vui lòng kiểm tra lại reference text có khớp không...")

        return processed_ref_audio

    def process_ref_text(ref_audio, ref_text, show_info=gr.Info):
        if not ref_text.strip():
            show_info("Vì reference text không được cấp, sẽ tiến hành dịch reference audio...")
            processed_ref_text = process_text(transcribe(ref_audio))
        else:
            show_info("Dùng reference text mà bạn cấp...")
            processed_ref_text = process_text(ref_text)
        processed_ref_text += ' '
        return processed_ref_text

    def save_voice_configs(voice_lang, voice_path, ref_text):
        voice_configs = "voice_configs.json"

        # Get base name of ref_audio as key
        voice_key = f"{voice_lang}_{os.path.basename(voice_path).split('.')[0]}"

        # Add or update voice entry
        global voices
        voices[voice_key] = {
            "ref_audio": voice_path,
            "ref_text": ref_text
        }
        
        # Save updated config
        try:
            with open(voice_configs, 'w', encoding='utf-8') as f:
                json.dump(voices, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving voice configuration: {str(e)}")
        
        gr.Info("Tạo giọng mới thành công")

        return "", "vi", None, None

    create_name_btn.click(process_voice_name, voice_name, voice_path)
    delete_name_btn.click(delete_voice, [voice_name, voice_lang], [voice_name, voice_lang])

    pp_audio_btn.click(
        process_ref_audio,
        inputs=[ref_audio, voice_path],
        outputs=[ref_audio],
    )

    pp_ref_text.click(
        process_ref_text,
        inputs=[ref_audio, ref_text],
        outputs=[ref_text],
    )

    create_voice_btn.click(
        save_voice_configs,
        inputs=[voice_lang, voice_path, ref_text],
        outputs=[voice_name, voice_lang, ref_text, ref_audio],
    )

with gr.Blocks() as app_tts:
    gr.Markdown("# TTS cơ bản")
    voice_option = gr.Dropdown(
        choices=list(voices.keys()) if voices else [],
        value="vi_dinh_soan",
        allow_custom_value=True,
        label="Chọn giọng",
        visible=True,
        interactive=bool(voices),  # Disable dropdown if voices is empty
        info="No voices available. Please create a new voice in the 'Tạo giọng' tab." if not voices else None
    )
    refresh_voice_btn = gr.Button("Làm mới giọng", variant="secondary")
    voice_ref_audio = gr.State("voices/dinh_soan/dinh_soan.wav")
    voice_ref_text = gr.State("Dựng đứng hết cả dậy , như thể âm hồn từ cõi giới u minh vừa thức tỉnh , vừa trở lại dương gian . ")

    gen_text_input = gr.Textbox(label="Text cần đọc", lines=10)
    gen_btn = gr.Button("Tạo", variant="primary")
    with gr.Accordion("Cấu hình nâng cao", open=False):
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
            value=False,
        )
        speed_slider = gr.Slider(
            label="Speed",
            minimum=0.3,
            maximum=2.0,
            value=1.0,
            step=0.1,
            info="Adjust the speed of the audio.",
        )
        nfe_slider = gr.Slider(
            label="NFE Steps",
            minimum=4,
            maximum=64,
            value=32,
            step=2,
            info="Set the number of denoising steps.",
        )
        cross_fade_duration_slider = gr.Slider(
            label="Cross-Fade Duration (s)",
            minimum=0.0,
            maximum=1.0,
            value=0.15,
            step=0.01,
            info="Set the duration of the cross-fade between audio clips.",
        )

    audio_output = gr.Audio(label="Synthesized Audio", show_download_button=True)
    spectrogram_output = gr.Image(label="Spectrogram")

    @gpu_decorator
    def basic_tts(
        voice_ref_audio,
        voice_ref_text,
        gen_text_input,
        remove_silence,
        cross_fade_duration_slider,
        nfe_slider,
        speed_slider,
    ):
        audio_out, spectrogram_path = infer(
            voice_ref_audio,
            voice_ref_text,
            gen_text_input,
            tts_model_choice,
            remove_silence,
            cross_fade_duration=cross_fade_duration_slider,
            nfe_step=nfe_slider,
            speed=speed_slider,
        )
        return audio_out, spectrogram_path
    
    def handle_change_voice(voice_option):
        if not voice_option:
            return gr.update(), gr.update(), gr.update()
        voice_ref_audio = os.path.abspath(voices[voice_option]["ref_audio"])
        voice_ref_text = voices[voice_option]["ref_text"]
        return gr.update(), voice_ref_audio, voice_ref_text

    def handle_refresh_voice():
        return gr.Dropdown(
            choices=list(voices.keys()) if voices else [],
            value="vi_dinh_soan",
            allow_custom_value=True,
            label="Choose Your Voice",
            visible=True,
            interactive=bool(voices),  # Disable dropdown if voices is empty
            info="No voices available. Please create a new voice in the 'Tạo giọng' tab." if not voices else None
        )

    refresh_voice_btn.click(handle_refresh_voice, None, voice_option)

    voice_option.change(
        handle_change_voice,
        inputs=[voice_option],
        outputs=[voice_option, voice_ref_audio, voice_ref_text]
    )

    gen_btn.click(
        basic_tts,
        inputs=[
            voice_ref_audio,
            voice_ref_text,
            gen_text_input,
            remove_silence,
            cross_fade_duration_slider,
            nfe_slider,
            speed_slider
        ],
        outputs=[audio_output, spectrogram_output],
    )

def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


with gr.Blocks() as app_multistyle:
    gr.Markdown(
        """
    # Tạo nhiều kiểu giọng nói khác nhau

    Tính năng này cho phép bạn tạo ra nhiều kiểu giọng (ví dụ: vui, buồn, tức giận...) hoặc nhiều giọng của các nhân vật khác nhau. Nhập văn bản theo định dạng bên dưới, hệ thống sẽ đọc bằng đúng kiểu giọng tương ứng. Nếu không ghi rõ, hệ thống sẽ mặc định dùng giọng thường. Giọng được giữ nguyên cho đến khi có kiểu mới được chỉ định.
    """
    )

    with gr.Row():
        gr.Markdown(
            """
            **Ví dụ 1:**                                                                      
            {BinhThuong} Xin chào, cho tôi gọi một phần bánh mì.                                                         
            {NgacNhien} Gì cơ, hết bánh mì rồi á?                                                                      
            {Buon} Tôi đã mong được ăn bánh mì mà...                                                              
            {TucGian} Biết gì không, tôi không quay lại cái tiệm này nữa!                                                                       
            {ThiTham} Thôi, tôi về nhà khóc tiếp vậy...                                                                           
            {LaHet} Tại sao lại là tôi?!                                                                         
            """
        )

        gr.Markdown(
            """
            **Ví dụ 2:**                                                                                
            {NhanVat1_Vui} Xin chào, cho tôi một phần bánh mì nhé.                                                            
            {NhanVat2_BinhThuong} Xin lỗi, tụi em hết bánh mì rồi.                                                                                
            {NhanVat1_Buon} Tôi đã rất mong chờ phần đó...                                                                             
            {NhanVat2_ThiTham} Em còn giấu một phần, để lấy cho anh.                                                                     
            """
        )

    gr.Markdown(
        "Tải lên file âm thanh khác nhau cho từng kiểu giọng nếu muốn huấn luyện thêm. Kiểu giọng đầu tiên là bắt buộc. Nhấn nút 'Thêm kiểu giọng' để thêm nhiều kiểu khác."
    )

    # Regular speech type (mandatory)
    with gr.Row() as regular_row:
        with gr.Column(scale=1):
            regular_name = gr.Textbox(value="DinhSoan", label="Tên kiểu giọng")
            regular_insert = gr.Button("Thêm nhãn", variant="secondary")
        with gr.Column(scale=1):
            regular_voice_option = gr.Dropdown(
                choices=list(voices.keys()) if voices else [],
                value="vi_dinh_soan",
                allow_custom_value=True,
                label="Chọn giọng",
                visible=True,
                interactive=bool(voices),  # Disable dropdown if voices is empty
                info="No voices available. Please create a new voice in the 'Tạo giọng' tab." if not voices else None
            )
            refresh_voice_btn = gr.Button("Làm mới giọng", variant="secondary")
            refresh_voice_btn.click(handle_refresh_voice, None, regular_voice_option)
        # regular_audio = gr.Audio(label="Regular Reference Audio", type="filepath")
        # regular_ref_text = gr.Textbox(label="Reference Text (Regular)", lines=2)

    # Regular speech type (max 100)
    max_speech_types = 100
    speech_type_rows = [regular_row]
    speech_type_names = [regular_name]
    speech_type_options = [regular_voice_option]
    speech_type_delete_btns = [None]
    speech_type_insert_btns = [regular_insert]
    speech_type_values = ["vi_dinh_soan"]

    # Additional speech types (99 more)
    for i in range(max_speech_types - 1):
        with gr.Row(visible=False) as row:
            with gr.Column(scale=1):
                name_input = gr.Textbox(label="Tên kiểu giọng")
                delete_btn = gr.Button("Xóa kiểu giọng", variant="secondary")
                insert_btn = gr.Button("Thêm nhãn", variant="secondary")
            with gr.Column(scale=1):
                voice_option = gr.Dropdown(
                    choices=list(voices.keys()) if voices else [],
                    value="",
                    allow_custom_value=True,
                    label="Chọn giọng",
                    visible=True,
                    interactive=bool(voices),  # Disable dropdown if voices is empty
                    info="No voices available. Please create a new voice in the 'Tạo giọng' tab." if not voices else None
                )
                refresh_voice_btn = gr.Button("Làm mới giọng", variant="secondary")
                refresh_voice_btn.click(handle_refresh_voice, None, voice_option)
        speech_type_rows.append(row)
        speech_type_names.append(name_input)
        speech_type_options.append(voice_option)
        speech_type_delete_btns.append(delete_btn)
        speech_type_insert_btns.append(insert_btn)
        speech_type_values.append("")

    def handle_change_speech_type_option(i, speech_type_option):
        global speech_type_values
        speech_type_values[i] = speech_type_option if speech_type_option else ''

    for idx, speech_type_option in enumerate(speech_type_options):
        speech_type_option.change(
            lambda speech_type_option, i=idx: handle_change_speech_type_option(i, speech_type_option),
            inputs=[speech_type_option],
            outputs=None
        )

    # Button to add speech type
    add_speech_type_btn = gr.Button("Thêm kiểu giọng")

    # Keep track of autoincrement of speech types, no roll back
    speech_type_count = 1

    # Function to add a speech type
    def add_speech_type_fn():
        row_updates = [gr.update() for _ in range(max_speech_types)]
        global speech_type_count
        if speech_type_count < max_speech_types:
            row_updates[speech_type_count] = gr.update(visible=True)
            speech_type_count += 1
        else:
            gr.Warning("Số lượng kiểu giọng cho phép đã hết. Bạn nên restart lại app.")
        return row_updates
    
    add_speech_type_btn.click(add_speech_type_fn, outputs=speech_type_rows)

    # Function to delete a speech type
    def delete_speech_type_fn():
        global speech_type_count
        speech_type_count -= 1
        return gr.update(visible=False), None, None

    # Update delete button clicks
    for i in range(1, len(speech_type_delete_btns)):
        speech_type_delete_btns[i].click(
            delete_speech_type_fn,
            outputs=[speech_type_rows[i], speech_type_names[i], speech_type_options[i]],
        )

    # Text input for the prompt
    gen_text_input_multistyle = gr.Textbox(
        label="Text cần đọc",
        lines=10,
        placeholder="Enter the script with speaker names (or emotion types) at the start of each block, e.g.:\n\n{Regular} Hello, I'd like to order a sandwich please.\n{Surprised} What do you mean you're out of bread?\n{Sad} I really wanted a sandwich though...\n{Angry} You know what, darn you and your little shop!\n{Whisper} I'll just go back home and cry now.\n{Shouting} Why me?!",
    )

    def make_insert_speech_type_fn(index):
        def insert_speech_type_fn(current_text, speech_type_name):
            current_text = current_text or ""
            speech_type_name = speech_type_name or "None"
            updated_text = current_text + f"{{{speech_type_name}}} "
            return updated_text

        return insert_speech_type_fn

    for i, insert_btn in enumerate(speech_type_insert_btns):
        insert_fn = make_insert_speech_type_fn(i)
        insert_btn.click(
            insert_fn,
            inputs=[gen_text_input_multistyle, speech_type_names[i]],
            outputs=gen_text_input_multistyle,
        )

    with gr.Accordion("Cấu hình nâng cao", open=False):
        remove_silence_multistyle = gr.Checkbox(
            label="Remove Silences",
            value=True,
        )

    # Generate button
    generate_multistyle_btn = gr.Button("Tạo", variant="primary")

    # Output audio
    audio_output_multistyle = gr.Audio(label="Synthesized Audio")

    @gpu_decorator
    def generate_multistyle_speech(
        gen_text,
        *args,
    ):
        speech_type_names_list = args[:max_speech_types]
        speech_type_options_list = args[max_speech_types : 2 * max_speech_types]
        speech_type_audios_list = [os.path.abspath(voices[speech_type_option]["ref_audio"]) if speech_type_option and speech_type_option in voices else None for speech_type_option in speech_type_options_list]
        speech_type_ref_texts_list = [voices[speech_type_option]["ref_text"] if speech_type_option and speech_type_option in voices else None for speech_type_option in speech_type_options_list]
        remove_silence = args[2 * max_speech_types : 3 * max_speech_types]
        speech_types = OrderedDict()

        ref_text_idx = 0
        for name_input, audio_input, ref_text_input in zip(
            speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list
        ):
            if name_input and audio_input:
                speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input}
            else:
                speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": ""}
            ref_text_idx += 1

        # Parse the gen_text into segments
        segments = parse_speechtypes_text(gen_text)

        # For each segment, generate speech
        generated_audio_segments = []
        current_style = "DinhSoan"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
            else:
                gr.Warning(f"Kiểu giọng {style} không tồn tại nên sẽ dùng kiểu mặc định là DinhSoan.")
                current_style = "DinhSoan"

            try:
                ref_audio = speech_types[current_style]["audio"]
            except KeyError:
                gr.Warning(f"Kiểu giọng {current_style} bị lỗi!")
                return None
            ref_text = speech_types[current_style].get("ref_text", "")

            # Generate speech for this segment
            audio_out, _ = infer(
                ref_audio, ref_text, text, tts_model_choice, remove_silence, cross_fade_duration=0, show_info=print
            )  # show_info=print no pull to top when generating
            sr, audio_data = audio_out

            generated_audio_segments.append(audio_data)
            speech_types[current_style]["ref_text"] = ref_text

        # Concatenate all audio segments
        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)
            return (sr, final_audio_data)
        else:
            gr.Warning("Không có audio nào được tạo")
            return None

    generate_multistyle_btn.click(
        generate_multistyle_speech,
        inputs=[
            gen_text_input_multistyle,
        ]
        + speech_type_names
        + speech_type_options
        + [
            remove_silence_multistyle,
        ],
        outputs=audio_output_multistyle,
    )

    # Validation function to disable Generate button if speech types are missing
    def validate_speech_types(gen_text, regular_name, *args):
        speech_type_names_list = args

        # Collect the speech types names
        speech_types_available = set()
        if regular_name:
            speech_types_available.add(regular_name)
        for name_input in speech_type_names_list:
            if name_input:
                speech_types_available.add(name_input)

        # Parse the gen_text to get the speech types used
        segments = parse_speechtypes_text(gen_text)
        speech_types_in_text = set(segment["style"] for segment in segments)

        # Check if all speech types in text are available
        missing_speech_types = speech_types_in_text - speech_types_available

        if missing_speech_types:
            # Disable the generate button
            return gr.update(interactive=False)
        else:
            # Enable the generate button
            return gr.update(interactive=True)

    gen_text_input_multistyle.change(
        validate_speech_types,
        inputs=[gen_text_input_multistyle, regular_name] + speech_type_names,
        outputs=generate_multistyle_btn,
    )


with gr.Blocks() as app_chat:
    gr.Markdown(
        """
# Voice Chat
Have a conversation with an AI using your reference voice! 
1. Upload a reference audio clip and optionally its transcript.
2. Load the chat model.
3. Record your message through your microphone.
4. The AI will respond using the reference voice.
"""
    )

    chat_model_name_list = [
        "Qwen/Qwen2.5-3B-Instruct",
        "microsoft/Phi-4-mini-instruct",
    ]

    @gpu_decorator
    def load_chat_model(chat_model_name):
        show_info = gr.Info
        global chat_model_state, chat_tokenizer_state
        if chat_model_state is not None:
            chat_model_state = None
            chat_tokenizer_state = None
            gc.collect()
            torch.cuda.empty_cache()

        show_info(f"Đang tải chat model: {chat_model_name}")
        chat_model_state = AutoModelForCausalLM.from_pretrained(chat_model_name, torch_dtype="auto", device_map="auto")
        chat_tokenizer_state = AutoTokenizer.from_pretrained(chat_model_name)
        show_info(f"Chat model {chat_model_name} tải thành công!")

        return gr.update(visible=False), gr.update(visible=True)

    if USING_SPACES:
        load_chat_model(chat_model_name_list[0])

    chat_model_name_input = gr.Dropdown(
        choices=chat_model_name_list,
        value=chat_model_name_list[0],
        label="Chat Model Name",
        info="Enter the name of a HuggingFace chat model",
        allow_custom_value=not USING_SPACES,
    )
    load_chat_model_btn = gr.Button("Load Chat Model", variant="primary", visible=not USING_SPACES)
    chat_interface_container = gr.Column(visible=USING_SPACES)

    chat_model_name_input.change(
        lambda: gr.update(visible=True),
        None,
        load_chat_model_btn,
        show_progress="hidden",
    )
    load_chat_model_btn.click(
        load_chat_model, inputs=[chat_model_name_input], outputs=[load_chat_model_btn, chat_interface_container]
    )

    with chat_interface_container:
        with gr.Row():
            with gr.Column():
                ref_audio_chat = gr.Audio(label="Reference Audio", type="filepath")
            with gr.Column():
                with gr.Accordion("Cấu hình nâng cao", open=False):
                    remove_silence_chat = gr.Checkbox(
                        label="Remove Silences",
                        value=True,
                    )
                    ref_text_chat = gr.Textbox(
                        label="Reference Text",
                        info="Optional: Leave blank to auto-transcribe",
                        lines=2,
                    )
                    system_prompt_chat = gr.Textbox(
                        label="System Prompt",
                        value="You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                        lines=2,
                    )

        chatbot_interface = gr.Chatbot(label="Conversation")

        with gr.Row():
            with gr.Column():
                audio_input_chat = gr.Microphone(
                    label="Speak your message",
                    type="filepath",
                )
                audio_output_chat = gr.Audio(autoplay=True)
            with gr.Column():
                text_input_chat = gr.Textbox(
                    label="Type your message",
                    lines=1,
                )
                send_btn_chat = gr.Button("Send Message")
                clear_btn_chat = gr.Button("Clear Conversation")

        conversation_state = gr.State(
            value=[
                {
                    "role": "system",
                    "content": "You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                }
            ]
        )

        # Modify process_audio_input to use model and tokenizer from state
        @gpu_decorator
        def process_audio_input(audio_path, text, history, conv_state):
            """Handle audio or text input from user"""

            if not audio_path and not text.strip():
                return history, conv_state, ""

            if audio_path:
                text = process_text(text)

            if not text.strip():
                return history, conv_state, ""

            conv_state.append({"role": "user", "content": text})
            history.append((text, None))

            response = generate_response(conv_state, chat_model_state, chat_tokenizer_state)

            conv_state.append({"role": "assistant", "content": response})
            history[-1] = (text, response)

            return history, conv_state, ""

        @gpu_decorator
        def generate_audio_response(history, ref_audio, ref_text, remove_silence):
            """Generate TTS audio for AI response"""
            if not history or not ref_audio:
                return None

            last_user_message, last_ai_response = history[-1]
            if not last_ai_response:
                return None

            audio_result, _, ref_text_out = infer(
                ref_audio,
                ref_text,
                last_ai_response,
                tts_model_choice,
                remove_silence,
                cross_fade_duration=0.15,
                speed=1.0,
                show_info=print,  # show_info=print no pull to top when generating
            )
            return audio_result, ref_text_out

        def clear_conversation():
            """Reset the conversation"""
            return [], [
                {
                    "role": "system",
                    "content": "You are not an AI assistant, you are whoever the user says you are. You must stay in character. Keep your responses concise since they will be spoken out loud.",
                }
            ]

        def update_system_prompt(new_prompt):
            """Update the system prompt and reset the conversation"""
            new_conv_state = [{"role": "system", "content": new_prompt}]
            return [], new_conv_state

        # Handle audio input
        audio_input_chat.stop_recording(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            audio_input_chat,
        )

        # Handle text input
        text_input_chat.submit(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle send button
        send_btn_chat.click(
            process_audio_input,
            inputs=[audio_input_chat, text_input_chat, chatbot_interface, conversation_state],
            outputs=[chatbot_interface, conversation_state],
        ).then(
            generate_audio_response,
            inputs=[chatbot_interface, ref_audio_chat, ref_text_chat, remove_silence_chat],
            outputs=[audio_output_chat, ref_text_chat],
        ).then(
            lambda: None,
            None,
            text_input_chat,
        )

        # Handle clear button
        clear_btn_chat.click(
            clear_conversation,
            outputs=[chatbot_interface, conversation_state],
        )

        # Handle system prompt change and reset conversation
        system_prompt_chat.change(
            update_system_prompt,
            inputs=system_prompt_chat,
            outputs=[chatbot_interface, conversation_state],
        )


with gr.Blocks() as app:
    gr.Markdown(
        f"""
# 🔥 F5-ViX: TTS tiếng Việt nâng cấp cho dân chơi

F5-ViX = F5 TTS gốc + finetune hơn 1000h tiếng Việt (capleaf/viVoice).
Auto dịch text bằng Whisper nếu để trống. Tạo voice trước rồi xài. Khi tạo, ref audio >10s sẽ bị cắt và phải nhập lại ref text nếu ref audio khác với ban đầu.

**Tips xài ngon:**

* Ref audio 5-10s là đẹp, tối ưu 8-9s.
* Giọng phải rõ, đều, không quá nhanh, không cà giựt.
* Muốn đọc tiếng Anh → chọn `F5-TTS_v1`.
* Muốn đọc tiếng Việt → chọn model ở mục **Custom**.

Dễ xài, chất lượng xịn, test liền đi còn chờ gì.

""")

    last_used_custom = files("f5_tts").joinpath("infer/.cache/last_used_custom_model_info_v1.txt")

    def load_last_used_custom():
        try:
            custom = []
            with open(last_used_custom, "r", encoding="utf-8") as f:
                for line in f:
                    custom.append(line.strip())
            return custom
        except FileNotFoundError:
            last_used_custom.parent.mkdir(parents=True, exist_ok=True)
            return DEFAULT_TTS_MODEL_CFG

    def switch_tts_model(new_choice):
        global tts_model_choice
        if new_choice == "Custom":  # override in case webpage is refreshed
            custom_model_version, custom_ckpt_path, custom_vocab_path, custom_model_cfg = load_last_used_custom()
            tts_model_choice = ["Custom", custom_model_version, custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
            return (
                gr.update(visible=True, value=custom_model_version),
                gr.update(visible=True, value=custom_ckpt_path),
                gr.update(visible=True, value=custom_vocab_path),
                gr.update(visible=True, value=custom_model_cfg),
            )
        else:
            tts_model_choice = new_choice
            return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    def set_custom_model(custom_model_version, custom_ckpt_path, custom_vocab_path, custom_model_cfg):
        global tts_model_choice
        tts_model_choice = ["Custom", custom_model_version, custom_ckpt_path, custom_vocab_path, json.loads(custom_model_cfg)]
        with open(last_used_custom, "w", encoding="utf-8") as f:
            f.write(custom_model_version + "\n" + custom_ckpt_path + "\n" + custom_vocab_path + "\n" + custom_model_cfg + "\n")

    with gr.Row():
        if not USING_SPACES:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS", "Custom"], label="Chọn model", value=DEFAULT_TTS_MODEL
            )
        else:
            choose_tts_model = gr.Radio(
                choices=[DEFAULT_TTS_MODEL, "E2-TTS"], label="Chọn model", value=DEFAULT_TTS_MODEL
            )
        custom_model_version = gr.Dropdown(
            choices=["DiT_v0", "DiT_v1"],
            value=load_last_used_custom()[0],
            allow_custom_value=True,
            label="Phiên bản DiT",
            visible=False,
        )
        custom_ckpt_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[0]],
            value=load_last_used_custom()[1],
            allow_custom_value=True,
            label="Model: local_path | hf://user_id/repo_id/model_ckpt",
            visible=False,
        )
        custom_vocab_path = gr.Dropdown(
            choices=[DEFAULT_TTS_MODEL_CFG[1]],
            value=load_last_used_custom()[2],
            allow_custom_value=True,
            label="Vocab: local_path | hf://user_id/repo_id/vocab_file",
            visible=False,
        )
        custom_model_cfg = gr.Dropdown(
            choices=[
                DEFAULT_TTS_MODEL_CFG[2],
                json.dumps(
                    dict(
                        dim=1024,
                        depth=22,
                        heads=16,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
                json.dumps(
                    dict(
                        dim=768,
                        depth=18,
                        heads=12,
                        ff_mult=2,
                        text_dim=512,
                        text_mask_padding=False,
                        conv_layers=4,
                        pe_attn_head=1,
                    )
                ),
            ],
            value=load_last_used_custom()[3],
            allow_custom_value=True,
            label="Config: in a dictionary form",
            visible=False,
        )

    choose_tts_model.change(
        switch_tts_model,
        inputs=[choose_tts_model],
        outputs=[custom_model_version, custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_version.change(
        set_custom_model,
        inputs=[custom_model_version, custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_ckpt_path.change(
        set_custom_model,
        inputs=[custom_model_version, custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_vocab_path.change(
        set_custom_model,
        inputs=[custom_model_version, custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )
    custom_model_cfg.change(
        set_custom_model,
        inputs=[custom_model_version, custom_ckpt_path, custom_vocab_path, custom_model_cfg],
        show_progress="hidden",
    )

    gr.TabbedInterface(
        [app_tts, app_multistyle, app_chat, app_new_voice],
        ["TTS", "TTS đa giọng", "Chat giọng", "Tạo/Xóa giọng"]
    )

@click.command()
@click.option("--port", "-p", default=None, type=int, help="Port to run the app on")
@click.option("--host", "-H", default=None, help="Host to run the app on")
@click.option(
    "--share",
    "-s",
    default=False,
    is_flag=True,
    help="Share the app via Gradio share link",
)
@click.option("--api", "-a", default=True, is_flag=True, help="Allow API access")
@click.option(
    "--root_path",
    "-r",
    default=None,
    type=str,
    help='The root path (or "mount point") of the application, if it\'s not served from the root ("/") of the domain. Often used when the application is behind a reverse proxy that forwards requests to the application, e.g. set "/myapp" or full URL for application served at "https://example.com/myapp".',
)
@click.option(
    "--inbrowser",
    "-i",
    is_flag=True,
    default=False,
    help="Automatically launch the interface in the default web browser",
)

def main(port, host, share, api, root_path, inbrowser):
    try:
        global app
        if not USING_SPACES:
            print("Starting app...")
            app.queue(api_open=api).launch(
                server_name=host,
                server_port=port,
                share=share,
                show_api=api,
                root_path=root_path,
                inbrowser=inbrowser,
            )
        else:
            app.queue().launch()
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        cleanup()

def cleanup():
    print("Clean up ...")
    tmp_dir = "/tmp"
    gradio_dir = "/tmp/gradio"

    # Clean up preprocess temps
    for file_path in glob.glob(os.path.join(tmp_dir, "tmp*.wav")) + glob.glob(os.path.join(tmp_dir, "tmp*.png")):
        os.remove(file_path)

    # Clean up gradio temps
    for item in os.listdir(gradio_dir):
        item_path = os.path.join(gradio_dir, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Remove files or symbolic links
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Remove directories


if __name__ == "__main__":
    main()
