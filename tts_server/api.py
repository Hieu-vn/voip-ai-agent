import os
from fastapi import FastAPI
from pydantic import BaseModel
import soundfile as sf
import io
from nemo.collections.tts.models import FastPitchModel
from nemo.collections.tts.models import HifiGanModel
import torch
from starlette.responses import Response

app = FastAPI()
fastpitch = FastPitchModel.from_pretrained(model_name=os.getenv("FASTPITCH_MODEL_NAME", "nvidia/tts_en_fastpitch"), map_location="cuda")
vocoder = HifiGanModel.from_pretrained(model_name=os.getenv("VOCODER_MODEL_NAME", "nvidia/tts_hifigan"), map_location="cuda")
fastpitch.eval(); vocoder.eval()

class Req(BaseModel):
    text: str
    spk: str = "female_vi"
    rate: float = 1.0

@app.post("/synthesize")
def synthesize(req: Req):
    with torch.inference_mode():
        tokens, _, _ = fastpitch.parse(req.text)
        mel, _ = fastpitch.generate_spectrogram(tokens, pace=1.0/req.rate)
        audio = vocoder.convert_spectrogram_to_audio(mel)
        wav = audio.to("cpu").numpy().squeeze()
        buf = io.BytesIO()
        sf.write(buf, wav, 22050, format="WAV")
        return Response(buf.getvalue(), media_type="audio/wav")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
