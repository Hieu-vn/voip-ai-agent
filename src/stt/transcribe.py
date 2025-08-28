
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

def transcribe(audio_path: str) -> str:
    """
    Transcribes an audio file using the Whisper model.

    Args:
        audio_path: Path to the audio file.

    Returns:
        The transcribed text.
    """
    result = pipe(audio_path)
    return result["text"]

if __name__ == "__main__":
    from datasets import load_dataset
    import soundfile as sf

    # Load the VIVOS dataset
    dataset = load_dataset("vivos", split="train")
    sample = dataset[0]["audio"]

    # Save the audio sample to a file
    audio_path = "sample.wav"
    sf.write(audio_path, sample["array"], sample["sampling_rate"])

    # Transcribe the audio file
    text = transcribe(audio_path)
    print(f"Transcription: {text}")

