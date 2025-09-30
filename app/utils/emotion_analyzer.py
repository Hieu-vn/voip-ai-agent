"""
Emotion and Sentiment Analysis utility.

This module uses a pre-trained model from Hugging Face to analyze the sentiment
of a given text, which can be used to generate more empathetic and appropriate
responses from the LLM.
"""
import torch
import structlog
from transformers import pipeline, Pipeline
from typing import Optional

log = structlog.get_logger()

class EmotionAnalyzer:
    """A wrapper for a Hugging Face sentiment analysis pipeline."""

    def __init__(self, model_name: str = "wonrax/phobert-base-vietnamese-sentiment"):
        """
        Initializes and loads the sentiment analysis model.

        Args:
            model_name: The name of the Hugging Face model to use.
        """
        self.model_name = model_name
        self.pipeline: Optional[Pipeline] = None
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=device
            )
            log.info("Emotion analysis pipeline loaded successfully.", model=model_name, device=device)
        except Exception as e:
            log.error("Failed to load emotion analysis model.", model=model_name, exc_info=e)

    def analyze(self, text: str) -> str:
        """
        Analyzes the sentiment of the text.

        Args:
            text: The input text.

        Returns:
            A sentiment label ('positive', 'negative', or 'neutral').
        """
        if not self.pipeline or not text.strip():
            return "neutral"

        try:
            result = self.pipeline(text)[0]
            label = result['label']
            
            # Map labels from the specific model to a common format
            if label == "POS":
                return "positive"
            elif label == "NEG":
                return "negative"
            else: # NEU
                return "neutral"
        except Exception as e:
            log.warning("Error during emotion analysis, returning neutral.", exc_info=e)
            return "neutral"
