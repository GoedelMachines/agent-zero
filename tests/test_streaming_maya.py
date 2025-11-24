"""
Maya-1-Voice CPU Streaming Inference - Standalone Reference Implementation
Adapted for CPU usage using Hugging Face Transformers (removing VLLM dependency).
"""

import torch
import numpy as np
import asyncio
import threading
import queue
from typing import List, Optional, AsyncGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.streamers import BaseStreamer
from snac import SNAC

# ============================================================================
# CONSTANTS
# ============================================================================

CODE_START_TOKEN_ID = 128257  # Start of Speech (SOS)
CODE_END_TOKEN_ID = 128258    # End of Speech (EOS)
CODE_TOKEN_OFFSET = 128266    # Start of SNAC codes
SNAC_MIN_ID = 128266
SNAC_MAX_ID = 156937
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
SNAC_TOKENS_PER_FRAME = 7
DEFAULT_MIN_TOKENS = 28

# ============================================================================
# HELPERS: TOKEN STREAMER
# ============================================================================

class TokenQueueStreamer(BaseStreamer):
    """
    Custom streamer that puts generated tokens into a thread-safe queue.
    This allows the generation loop to run in a background thread while
    the main thread consumes tokens for audio decoding.
    """
    def __init__(self, token_queue: queue.Queue):
        self.token_queue = token_queue

    def put(self, value):
        # Value is usually a tensor of shape (batch_size, 1)
        # We assume batch_size=1 for this implementation
        if isinstance(value, torch.Tensor):
            value = value.cpu().tolist()
        
        # Flatten and put into queue
        if isinstance(value, list):
            for item in value:
                # Handle nested lists if batch > 1
                if isinstance(item, list):
                    for subitem in item:
                        self.token_queue.put(subitem)
                else:
                    self.token_queue.put(item)
        else:
            self.token_queue.put(value)

    def end(self):
        self.token_queue.put(None)  # Sentinel value to signal end

# ============================================================================
# SNAC DECODER (CPU Optimized)
# ============================================================================

class SNACDecoder:
    def __init__(self, device: str = "cpu"):
        self.device = device
        print(f"🎵 Loading SNAC 24kHz model to {device}...")
        # Force SNAC to CPU
        self.snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval().to(device)
        print(f"✅ SNAC decoder initialized")
    
    def unpack_snac_from_7(self, vocab_ids: List[int]) -> List[List[int]]:
        if vocab_ids and vocab_ids[-1] == CODE_END_TOKEN_ID:
            vocab_ids = vocab_ids[:-1]
        
        frames = len(vocab_ids) // SNAC_TOKENS_PER_FRAME
        vocab_ids = vocab_ids[:frames * SNAC_TOKENS_PER_FRAME]
        
        if frames == 0:
            return [[], [], []]
        
        l1, l2, l3 = [], [], []
        
        for i in range(frames):
            slots = vocab_ids[i*7:(i+1)*7]
            l1.append((slots[0] - CODE_TOKEN_OFFSET) % 4096)
            l2.extend([(slots[1] - CODE_TOKEN_OFFSET) % 4096, (slots[4] - CODE_TOKEN_OFFSET) % 4096])
            l3.extend([
                (slots[2] - CODE_TOKEN_OFFSET) % 4096, (slots[3] - CODE_TOKEN_OFFSET) % 4096,
                (slots[5] - CODE_TOKEN_OFFSET) % 4096, (slots[6] - CODE_TOKEN_OFFSET) % 4096,
            ])
        
        return [l1, l2, l3]
    
    @torch.inference_mode()
    def decode_to_bytes(self, snac_tokens: List[int], use_sliding_window: bool = False) -> Optional[bytes]:
        if len(snac_tokens) < SNAC_TOKENS_PER_FRAME:
            return None
        
        levels = self.unpack_snac_from_7(snac_tokens)
        if not levels[0]: return None
        
        # Convert to tensors on CPU
        codes = [torch.tensor(level, dtype=torch.long, device=self.device).unsqueeze(0) for level in levels]
        
        z_q = self.snac_model.quantizer.from_codes(codes)
        audio = self.snac_model.decoder(z_q)
        audio = audio[0, 0].cpu().numpy()
        
        if use_sliding_window and len(audio) >= 4096:
            audio = audio[2048:4096]
        
        return (audio * 32767).astype(np.int16).tobytes()

# ============================================================================
# MAYA-1-VOICE MODEL (Transformers CPU Implementation)
# ============================================================================

class Maya1VoiceModel:
    def __init__(self, model_path: str):
        print(f"🚀 Initializing Maya-1-Voice Model (CPU Mode)")
        
        # CPU Optimization: Use float32 for stability, or bfloat16 if CPU supports it
        # Explicitly setting device_map to cpu
        self.device = "cpu"
        
        print(f"📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        print(f"🧠 Loading model to CPU (this may take a moment)...")
        # We remove device_map="auto" to force CPU manually to be safe
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32, # Safe for CPU. Use torch.bfloat16 if you have a modern CPU
            trust_remote_code=True,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()
        
        print(f"✅ Model loaded")

    def build_prompt(self, description: str, text: str) -> str:
        content = f'<description="{description}"> {text}'
        messages = [{"role": "user", "content": content}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ============================================================================
# STREAMING PIPELINE
# ============================================================================

class Maya1VoiceStreamingPipeline:
    def __init__(self, model: Maya1VoiceModel, snac_decoder: SNACDecoder):
        self.model = model
        self.snac_decoder = snac_decoder

    async def generate_speech_stream(
        self, description: str, text: str, temperature: float = 0.4, 
        max_tokens: int = 2000, repetition_penalty: float = 1.1
    ) -> AsyncGenerator[bytes, None]:
        
        print(f"\n🌊 Starting streaming generation (CPU)")
        prompt = self.model.build_prompt(description, text)
        inputs = self.model.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Thread communication
        token_queue = queue.Queue()
        streamer = TokenQueueStreamer(token_queue)
        
        generation_kwargs = dict(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            streamer=streamer,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=repetition_penalty,
            eos_token_id=CODE_END_TOKEN_ID,
            pad_token_id=self.model.tokenizer.eos_token_id
        )

        # Run generation in a separate thread so we can yield bytes in this thread
        thread = threading.Thread(target=self.model.model.generate, kwargs=generation_kwargs)
        thread.start()

        token_buffer = []
        total_chunks = 0
        
        # Consume the queue
        while True:
            try:
                # Non-blocking check with small timeout to allow async loop to breathe
                # We use a blocking get with timeout to make it friendly to the loop
                token_id = token_queue.get(timeout=0.05)
            except queue.Empty:
                if not thread.is_alive():
                    break
                # Give control back to event loop briefly
                await asyncio.sleep(0.01)
                continue

            if token_id is None: # End sentinel
                break
                
            # Filter SNAC tokens
            if SNAC_MIN_ID <= token_id <= SNAC_MAX_ID:
                token_buffer.append(token_id)
                
                # Sliding window logic (same as original)
                if len(token_buffer) % 7 == 0 and len(token_buffer) > 27:
                    window_tokens = token_buffer[-28:]
                    audio_bytes = self.snac_decoder.decode_to_bytes(window_tokens, use_sliding_window=True)
                    
                    if audio_bytes:
                        total_chunks += 1
                        if total_chunks == 1:
                            print(f"🎵 First chunk decoded")
                        yield audio_bytes

        thread.join()
        print(f"✅ Streaming complete")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    MODEL_PATH = "maya-research/maya1"
    
    # Force CPU
    DEVICE = "cpu"
    
    model = Maya1VoiceModel(model_path=MODEL_PATH)
    snac_decoder = SNACDecoder(device=DEVICE)
    pipeline = Maya1VoiceStreamingPipeline(model, snac_decoder)
    
    description = "Realistic male voice in the 30s age with american accent. Normal pitch."
    text = "This is a test running entirely on the CPU. It might be slower, but it works!"
    
    audio_chunks = []
    
    # Calculate time to measure CPU performance
    import time
    start_time = time.time()
    
    async for chunk in pipeline.generate_speech_stream(
        description=description,
        text=text,
        max_tokens=400
    ):
        audio_chunks.append(chunk)
        print(f"📦 Chunk received: {len(chunk)} bytes", end="\r")
        
    duration = time.time() - start_time
    print(f"\n⏱️ Generation took {duration:.2f} seconds")

    full_audio = b''.join(audio_chunks)
    
    try:
        import wave
        with wave.open("cpu_output.wav", 'wb') as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(full_audio)
        print("💾 Saved to cpu_output.wav")
    except ImportError:
        pass

if __name__ == "__main__":
    asyncio.run(main())