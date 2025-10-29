import sys
import os

sys.path.append("external/dinov3")

import torch
from transformers import AutoImageProcessor, AutoModel

model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
processor = AutoImageProcessor.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))
model = AutoModel.from_pretrained(model_name, token=os.getenv("HF_TOKEN"))


print("Model loaded successfully!")
