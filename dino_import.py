# src/dino_import.py

import os
from transformers import AutoImageProcessor, AutoModel

HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "facebook/dinov3-vits16-pretrain-lvd1689m"

if not HF_TOKEN:
    raise EnvironmentError("HF_TOKEN not found. Please set it or load from a .env file.")

def load_dinov3():
    """
    Load and return the DINOv3 processor and model.
    """
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModel.from_pretrained(MODEL_NAME, token=HF_TOKEN, output_attentions=True)
    model.eval()
    print("model loaded")
    return processor, model

