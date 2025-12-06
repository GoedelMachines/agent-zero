import base64
import io
import warnings
import asyncio
import time
import soundfile as sf
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snac import SNAC

from python.helpers import runtime
from python.helpers.print_style import PrintStyle
from python.helpers.notification import NotificationManager, NotificationType, NotificationPriority

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

_model = None
_tokenizer = None
_snac_model = None
is_updating_model = False


# Hardcoding the voice description for now
_voice_description = "Simple male voice. low and deep pitch, conversational pacing."

# These are some important constants for SNAC
CODE_START_TOKEN_ID = 128257
CODE_END_TOKEN_ID = 128258
CODE_TOKEN_OFFSET = 128266
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_TOKENS_PER_FRAME = 7

SOH_ID = 128259
EOH_ID = 128260
SOA_ID = 128261
BOS_ID = 128000
TEXT_EOT_ID = 128009


def build_prompt(tokenizer, description: str, text: str) -> str:
    """
    Taken directly from hugging face
    """
    soh_token = tokenizer.decode([SOH_ID])
    eoh_token = tokenizer.decode([EOH_ID])
    soa_token = tokenizer.decode([SOA_ID])
    sos_token = tokenizer.decode([CODE_START_TOKEN_ID])
    eot_token = tokenizer.decode([TEXT_EOT_ID])
    bos_token = tokenizer.bos_token
    
    formatted_text = f'<description="{description}"> {text}'
    
    prompt = (
        soh_token + bos_token + formatted_text + eot_token +
        eoh_token + soa_token + sos_token
    )
    return prompt

def extract_snac_codes(token_ids: list) -> list:
    """
    Taken directly from hugging face
    """
    try:
        eos_idx = token_ids.index(CODE_END_TOKEN_ID)
    except ValueError:
        eos_idx = len(token_ids)
    
    snac_codes = [
        token_id for token_id in token_ids[:eos_idx]
        if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID
    ]
    return snac_codes

def unpack_snac_from_7(snac_tokens: list) -> list:
    """
    Taken directly from hugging face
    """
    if snac_tokens and snac_tokens[-1] == CODE_END_TOKEN_ID:
        snac_tokens = snac_tokens[:-1]
    
    frames = len(snac_tokens) // SNAC_TOKENS_PER_FRAME
    snac_tokens = snac_tokens[:frames * SNAC_TOKENS_PER_FRAME]
    
    if frames == 0:
        return [[], [], []]
    
    l1, l2, l3 = [], [], []
    
    for i in range(frames):
        slots = snac_tokens[i*7:(i+1)*7]
        l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
        l2.extend([
            (slots[1] - CODE_TOKEN_OFFSET) % 4096,
            (slots[4] - CODE_TOKEN_OFFSET) % 4096,
        ])
        l3.extend([
            (slots[2] - CODE_TOKEN_OFFSET) % 4096,
            (slots[3] - CODE_TOKEN_OFFSET) % 4096,
            (slots[5] - CODE_TOKEN_OFFSET) % 4096,
            (slots[6] - CODE_TOKEN_OFFSET) % 4096,
        ])
    
    return [l1, l2, l3]


async def preload():
    try:
        # return await runtime.call_development_function(_preload)
        return await _preload()
    except Exception as e:
        # if not runtime.is_development():
        raise e
        # return await _preload()

async def _preload():
    global _model, _tokenizer, _snac_model, is_updating_model

    while is_updating_model:
        await asyncio.sleep(0.1)

    try:
        is_updating_model = True
        if not _model:
            NotificationManager.send_notification(
                NotificationType.INFO,
                NotificationPriority.NORMAL,
                "Loading Maya1 & SNAC models...",
                display_time=99,
                group="maya1-preload")
            PrintStyle.standard("Loading Maya1 & SNAC models...")
            
            # Determine Device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            PrintStyle.standard(f"Using device: {device}")

            # 1. Load Maya1
            _model = AutoModelForCausalLM.from_pretrained(
                "maya-research/maya1", 
                dtype=torch.bfloat16 if device == "cuda" else torch.float32, 
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            if device == "cpu":
                _model = _model.to(device)

            _tokenizer = AutoTokenizer.from_pretrained(
                "maya-research/maya1",
                trust_remote_code=True
            )

            # 2. Load SNAC
            _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
            _snac_model = _snac_model.to(device)

            NotificationManager.send_notification(
                NotificationType.INFO,
                NotificationPriority.NORMAL,
                "Maya1 TTS models loaded.",
                display_time=2,
                group="maya1-preload")
            
    except Exception as e:
        PrintStyle.error(f"Failed to load Maya1 models: {e}")
        raise e
    finally:
        is_updating_model = False


async def is_downloading():
    try:
        # return await runtime.call_development_function(_is_downloading)
        return _is_downloading()
    except Exception as e:
        # if not runtime.is_development():
        raise e
        # return _is_downloading()

def _is_downloading():
    return is_updating_model

async def is_downloaded():
    try:
        # return await runtime.call_development_function(_is_downloaded)
        return _is_downloaded()
    except Exception as e:
        # if not runtime.is_development():
        raise e
        # return _is_downloaded()

def _is_downloaded():
    return _model is not None and _snac_model is not None


async def synthesize_sentences(sentences: list[str]):
    """
    Exposes the same synthesize_sentences method as kokoro_tts
    """
    try:
        # return await runtime.call_development_function(_synthesize_sentences, sentences)
        return await _synthesize_sentences(sentences)
    except Exception as e:
        # if not runtime.is_development():
        raise e
        # return await _synthesize_sentences(sentences)


async def _synthesize_sentences(sentences: list[str]):
    await _preload()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    combined_audio = []

    try:
        for sentence in sentences:
            if not sentence.strip():
                continue

            # 1. Prepare Prompt
            prompt = build_prompt(_tokenizer, _voice_description, sentence.strip())
            
            inputs = _tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Generate Tokens (we can run in thread if blocking becomes an issue, but standard inference here)
            with torch.inference_mode():
                outputs = _model.generate(
                    **inputs, 
                    max_new_tokens=4096, 
                    min_new_tokens=28, 
                    temperature=0.1, 
                    top_p=0.9, 
                    repetition_penalty=1.1, 
                    do_sample=True,
                    eos_token_id=CODE_END_TOKEN_ID, 
                    pad_token_id=_tokenizer.pad_token_id,
                )

            # Process Output
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:].tolist()
            snac_tokens = extract_snac_codes(generated_ids)
            levels = unpack_snac_from_7(snac_tokens)


            # Check if we actually got audio frames
            if not levels or not levels[0]:
                PrintStyle.standard(f"No audio frames generated for sentence: {sentence[:20]}...")
                continue

            codes_tensor = [
                torch.tensor(level, dtype=torch.long, device=device).unsqueeze(0)
                for level in levels
            ]

            # 4. Decode to Audio
            with torch.inference_mode():
                z_q = _snac_model.quantizer.from_codes(codes_tensor)
                audio_tensor = _snac_model.decoder(z_q)[0, 0]
                audio_numpy = audio_tensor.detach().cpu().numpy()
                combined_audio.extend(audio_numpy)

        # 5. Convert combined audio to bytes
        if len(combined_audio) == 0:
            # Handle edge case where nothing was generated
            return ""

        buffer = io.BytesIO()
        sf.write(buffer, combined_audio, 24000, format="WAV")
        audio_bytes = buffer.getvalue()

        # Return base64 encoded audio, same as kokoro_tts
        return base64.b64encode(audio_bytes).decode("utf-8")

    except Exception as e:
        PrintStyle.error(f"Error in Maya1 TTS synthesis: {e}")
        import traceback
        traceback.print_exc()
        raise e