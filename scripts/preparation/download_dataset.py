#!/usr/bin/env python3
import argparse
import logging
from pathlib import Path
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def download_repo(
    repo_id: str,
    local_dir: Path,
    repo_type: str = "dataset",
    revision: str = "main",
    allow_patterns: list = None
):
    """Downloads a repository from the Hugging Face Hub."""
    local_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting download of '{repo_id}' (revision: {revision}) to '{local_dir}'...")
    logger.info("This may take a long time depending on the dataset size and your connection speed.")

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False, # Set to False to download actual files
            allow_patterns=allow_patterns or ["*.parquet"], # Default to only parquet files
            tqdm_class=None # Can be replaced with a custom tqdm class if needed
        )
        logger.info(f"Successfully downloaded '{repo_id}' to '{local_dir}'.")
    except Exception as e:
        logger.error(f"An error occurred during download: {e}", exc_info=True)
        raise

def main():
    parser = argparse.ArgumentParser(description="Download a dataset from the Hugging Face Hub.")
    parser.add_argument(
        "--repo_id", 
        default="thivux/phoaudiobook", 
        help="The ID of the repository on the Hugging Face Hub (e.g., 'thivux/phoaudiobook')."
    )
    parser.add_argument(
        "--local_dir", 
        required=True, 
        help="The local directory to save the dataset to."
    )
    parser.add_argument(
        "--repo_type", 
        default="dataset", 
        choices=["dataset", "model", "space"], 
        help="The type of the repository."
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="The specific revision (branch, tag, commit hash) to download."
    )
    args = parser.parse_args()

    download_repo(
        repo_id=args.repo_id,
        local_dir=Path(args.local_dir),
        repo_type=args.repo_type,
        revision=args.revision
    )

if __name__ == "__main__":
    main()