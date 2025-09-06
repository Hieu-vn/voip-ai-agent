
import os
from huggingface_hub import snapshot_download
from loguru import logger

HUGGING_FACE_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "unsloth/Llama-4-Maverick-17B-128E-Instruct"

def main():
    logger.info(f"Starting download for model: {MODEL_NAME}")
    logger.info("This process will download the model files to the local cache.")
    logger.info("It may take a significant amount of time depending on the model size and network speed.")
    
    try:
        snapshot_download(
            repo_id=MODEL_NAME,
            token=HUGGING_FACE_TOKEN,
            local_dir_use_symlinks=False # Set to False for better compatibility
        )
        logger.success(f"Successfully downloaded model '{MODEL_NAME}'.")
        logger.info("The model is now stored locally. We can now proceed to start the server.")
    except Exception as e:
        logger.error(f"An error occurred during the download: {e}")
        logger.error("Please double-check the model name and ensure your Hugging Face token has access.")

if __name__ == "__main__":
    main()
