# api/synthesize.py

from python.helpers.api import ApiHandler, Request, Response

# from python.helpers import runtime, settings, kokoro_tts
from python.helpers import runtime, settings, maya1_tts

# class Synthesize(ApiHandler):
#     async def process(self, input: dict, request: Request) -> dict | Response:
#         text = input.get("text", "")
#         ctxid = input.get("ctxid", "")
        
#         if ctxid:
#             context = self.use_context(ctxid)

#         try:
#             # audio is chunked on the frontend for better flow
#             audio = await kokoro_tts.synthesize_sentences([text])
#             return {"audio": audio, "success": True}
#         except Exception as e:
#             return {"error": str(e), "success": False}

class Synthesize(ApiHandler):
    async def process(self, input: dict, request: Request) -> dict | Response:
        text = input.get("text", "")
        ctxid = input.get("ctxid", "")
        
        if ctxid:
            context = self.use_context(ctxid)

        try:
            # audio is chunked on the frontend for better flow
            audio = await maya1_tts.synthesize_sentences([text])
            return {"audio": audio, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}