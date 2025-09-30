
import logging
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from nemo.collections.tts.models import FastPitchModel
from nemo.utils.exp_manager import exp_manager

logging.basicConfig(level=logging.INFO)

def main():
    # --- Load Configuration ---
    config_path = "/data/voip-ai-agent/config/training/config_finetune_vi.yaml"
    cfg = OmegaConf.load(config_path)
    
    # --- Setup Trainer & Experiment Manager ---
    # The experiment manager will create a directory for logs and checkpoints
    # The directory will be named based on the 'name' in the config file.
    trainer = Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # --- Initialize or Restore Model ---
    # We use restore_from because we are initializing from a .nemo file
    logging.info(f"Initializing model from pre-trained checkpoint: {cfg.model.init_from_nemo_model}")
    model = FastPitchModel.restore_from(cfg.model.init_from_nemo_model, override_config_path=config_path)
    
    # Set the new dataset configuration to the model
    model.setup_training_data(cfg.model.train_ds)
    model.setup_validation_data(cfg.model.validation_ds)

    # --- Start Fine-tuning ---
    logging.info("Starting fine-tuning...")
    trainer.fit(model)
    logging.info("Fine-tuning finished.")

    # --- Save the final model ---
    # The experiment manager will save checkpoints, but we can also save the final one manually.
    final_model_path = "/data/voip-ai-agent/models/tts/vi/FastPitch_Vietnamese_Finetuned.nemo"
    model.save_to(final_model_path)
    logging.info(f"Final fine-tuned model saved to: {final_model_path}")

if __name__ == '__main__':
    # Ensure the script is run with a PyTorch distributed launcher if using multi-GPU
    # Example: torchrun --nproc_per_node=8 run_finetune.py
    main()
