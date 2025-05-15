import os
import json
import tempfile
import re
import unicodedata
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
from datetime import datetime
import uvicorn

from collections import OrderedDict

import soundfile as sf
import torchaudio
from cached_path import cached_path

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


from f5_tts.model import DiT, DiT_v0
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    infer_process,
    remove_silence_for_generated_wav,
)

UPLOAD_DIR = "public"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load voice configurations from JSON file
try:
    with open('voice_configs.json', 'r', encoding='utf-8') as f:
        VOICE_CONFIGS = json.load(f)
except FileNotFoundError:
    print("Warning: voice_configs.json not found. Using default configuration.")
    VOICE_CONFIGS = {
        "vi_dinh_soan": {
            "ref_audio": "voices/dinh_soan/dinh_soan.wav",
            "ref_text": "Dựng đứng hết cả dậy , như thể âm hồn từ cõi giới u minh vừa thức tỉnh , vừa trở lại dương gian . "
        }
    }

DEFAULT_VOICE = "vi_dinh_soan"

DEFAULT_TTS_MODEL = "F5-TTS_v1"

DEFAULT_TTS_MODEL_CFG = [
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]

class TTSRequest(BaseModel):
    speed: float
    text: str
    voice: str

class TTSMultiSpeechRequest(BaseModel):
    speed: float
    text: str
    tags: List[str]
    voices: List[str]
    remove_silence: bool

vocoder = load_vocoder()

def load_f5tts():
    ckpt_path = str(cached_path(DEFAULT_TTS_MODEL_CFG[0]))
    F5TTS_model_cfg = json.loads(DEFAULT_TTS_MODEL_CFG[2])
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)

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

custom_ema_model, pre_custom_path = None, ""

def process_text(text):
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
def infer(
    ref_audio,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
    nfe_step=32,
    speed=1,
    show_info=print
):
    gen_text = process_text(gen_text)

    if model == DEFAULT_TTS_MODEL:
        ema_model = load_f5tts()
    elif isinstance(model, list) and model[0] == "Custom":
        assert not USING_SPACES, "Only official checkpoints allowed in Spaces."
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[2]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(model[1], model[2], vocab_path=model[3], model_cfg=model[4])
            pre_custom_path = model[2]
        ema_model = custom_ema_model

    final_wave, final_sample_rate, _ = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    return final_sample_rate, final_wave


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "DinhSoan"

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

@gpu_decorator
def generate_multistyle_speech(
    gen_text,
    speech_type_names_list,
    speech_type_options_list,
    remove_silence,
    speed
):
    speech_type_audios_list = []
    speech_type_ref_texts_list = []
    speech_type_lang_list = []
    for speech_type_option in speech_type_options_list:
        if speech_type_option and speech_type_option:
            speech_type_audios_list.append(os.path.abspath(VOICE_CONFIGS[speech_type_option]["ref_audio"]))
            speech_type_ref_texts_list.append(VOICE_CONFIGS[speech_type_option]["ref_text"])
            speech_type_lang_list.append(speech_type_option[:2])
        else:
            speech_type_audios_list.append(None)
            speech_type_ref_texts_list.append(None)
            speech_type_lang_list.append(None)

    speech_types = OrderedDict()

    ref_text_idx = 0
    for name_input, audio_input, ref_text_input, lang in zip(
        speech_type_names_list, speech_type_audios_list, speech_type_ref_texts_list, speech_type_lang_list
    ):
        if name_input and audio_input:
            speech_types[name_input] = {"audio": audio_input, "ref_text": ref_text_input, "lang": lang}
        else:
            speech_types[f"@{ref_text_idx}@"] = {"audio": "", "ref_text": "", "lang": ""}
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
            current_style = "DinhSoan"

        try:
            ref_audio = speech_types[current_style]["audio"]
        except KeyError:
            return None
        ref_text = speech_types[current_style].get("ref_text", "")
        lang = speech_types[current_style].get("lang", "")

        if lang == "vi":
            model_ver = "DiT_v1"
            ckpt_path = "/home/bobacu/Projects/TrainVoice/F5-TTS/ckpts/vivoice/DiT_v1/model_epoch_60.pt"
            vocab_path = "/home/bobacu/Projects/TrainVoice/F5-TTS/data/vivoice_char/DiT_v1/vocab.txt"
            model_cfg = {'dim': 1024, 'depth': 22, 'heads': 16, 'ff_mult': 2, 'text_dim': 512, 'conv_layers': 4}
            ema_model = ["Custom", model_ver, ckpt_path, vocab_path, model_cfg]
        elif lang in ["en", "zh"]:
            ema_model = DEFAULT_TTS_MODEL

        # Generate speech for this segment
        sr, audio_data = infer(
            ref_audio, ref_text, text, ema_model, remove_silence=remove_silence, speed=speed, cross_fade_duration=0, show_info=print
        )

        generated_audio_segments.append(audio_data)
        speech_types[current_style]["ref_text"] = ref_text

    # Concatenate all audio segments
    if generated_audio_segments:
        final_audio_data = np.concatenate(generated_audio_segments)
        return (sr, final_audio_data)
    else:
        return None

def cleanup():
    print("Clean up ...")
    public_dir = "./public"

    # Clean up preprocess temps
    for file_name in os.listdir(public_dir):
        if file_name.endswith(".wav"):
            os.remove(os.path.join(public_dir, file_name))

app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")

@app.get("/welcome")
async def sayHello():
    return {"message": "Hello World"}

@app.post("/tts")
async def text_to_speech(request: TTSRequest, request_obj: Request):
    try:
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tts_{timestamp}.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        
        # Get voice configuration
        voice_config = VOICE_CONFIGS.get(request.voice)

        ref_audio_input = voice_config["ref_audio"]
        ref_text_input = voice_config["ref_text"]
        
        # if voice contain "en-" then use e2tts model
        if "vi_" in request.voice:
            model_ver = "DiT_v1"
            ckpt_path = "/home/bobacu/Projects/TrainVoice/F5-TTS/ckpts/vivoice/DiT_v1/model_epoch_60.pt"
            vocab_path = "/home/bobacu/Projects/TrainVoice/F5-TTS/data/vivoice_char/DiT_v1/vocab.txt"
            model_cfg = {'dim': 1024, 'depth': 22, 'heads': 16, 'ff_mult': 2, 'text_dim': 512, 'conv_layers': 4}
            ema_model = ["Custom", model_ver, ckpt_path, vocab_path, model_cfg]
        else:
            ema_model = DEFAULT_TTS_MODEL
        
        # Generate speech
        audio_out = infer(
            ref_audio_input,
            ref_text_input,
            request.text,
            ema_model,
            remove_silence=False,
            cross_fade_duration=0.15,
            nfe_step=32,
            speed=request.speed,
        )
        
        # Extract audio data and sample rate
        sample_rate, audio_data = audio_out
        
        # Save the audio file
        sf.write(output_path, audio_data, sample_rate)
        
        # Get the hostname from the request
        hostname = str(request_obj.base_url).rstrip('/')
        
        return JSONResponse(
            content={
                "message": "Speech generated successfully",
                "audio_path": f"{hostname}/public/{output_filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts-multispeech")
async def text_to_speech(request: TTSMultiSpeechRequest, request_obj: Request):
    try:
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"tts_{timestamp}.wav"
        output_path = os.path.join(UPLOAD_DIR, output_filename)
        
        # Generate speech
        audio_out = generate_multistyle_speech(
            request.text,
            request.tags,
            request.voices,
            request.remove_silence,
            request.speed
        )
        
        # Extract audio data and sample rate
        sample_rate, audio_data = audio_out
        
        # Save the audio file
        sf.write(output_path, audio_data, sample_rate)
        
        # Get the hostname from the request
        hostname = str(request_obj.base_url).rstrip('/')
        
        return JSONResponse(
            content={
                "message": "Speech generated successfully",
                "audio_path": f"{hostname}/public/{output_filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    try:
        uvicorn.run("api:app", host="0.0.0.0", port=8003, reload=True)
    except Exception as e:
        print(f"Exception occurred: {e}")
    finally:
        cleanup()