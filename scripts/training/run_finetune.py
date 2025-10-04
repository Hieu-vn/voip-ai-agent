import logging
import torch
from omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from nemo.collections.tts.models import FastPitchModel
from nemo.utils.exp_manager import exp_manager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # --- 1. Load Configuration ---
    config_path = "/config/training/config_finetune_vi.yaml"
    cfg = OmegaConf.load(config_path)
    logging.info("Successfully loaded configuration from: %s", config_path)
    logging.info(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")
    # --- 2. Setup Trainer & Experiment Manager ---
    # The experiment manager creates a directory for logs and checkpoints
    trainer = Trainer(**cfg.trainer)
    exp_manager(trainer, cfg.get("exp_manager", None))

    # --- 3. Initialize Model from Pre-trained Checkpoint ---
    logging.info(f"Initializing model from pre-trained checkpoint: {cfg.model.init_from_nemo_model}")
    # Restore the model. Its internal config is from the original .nemo file for now.
    model = FastPitchModel.restore_from(cfg.model.init_from_nemo_model, trainer=trainer)
    
    # --- 4. Update Model Configuration for Fine-tuning ---
    # This is the crucial step: update the model's internal config with our new settings
    # for datasets, optimizer, and any other overrides.
    with open_dict(model.cfg):
        model.cfg.update(cfg.model)
    
    logging.info("Model configuration updated for fine-tuning.")
    logging.info(f"Optimizer config that will be used by NeMo:\n{OmegaConf.to_yaml(model.cfg.optim)}")
    # --- 5. Setup Data Loaders ---
    # The model uses its updated config (model.cfg) to set up the data loaders.
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    logging.info("Training and validation data loaders set up successfully.")

    # --- 6. Start Fine-tuning ---
    # The Trainer will call model.configure_optimizers(), which automatically uses the `optim` 
    # section from the now-updated `model.cfg`. No manual optimizer handling is needed.
    logging.info("Starting fine-tuning...")
    trainer.fit(model)
    logging.info("Fine-tuning finished.")

    # --- 7. Save the Final Model ---
    # The experiment manager saves checkpoints, but we can also save the final one manually.
    final_model_path = "/data/voip-ai-agent/models/tts/vi/FastPitch_Vietnamese_Finetuned.nemo"
    model.save_to(final_model_path)
    logging.info(f"Final fine-tuned model saved to: {final_model_path}")

if __name__ == '__main__':
    main()