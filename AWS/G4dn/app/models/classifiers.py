from transformers import pipeline
import torch

classifier = None
try:
    if torch.cuda.is_available():
        device_num = 0
        print(f"CUDA available. Loading BART model on cuda:{device_num}")
    else:
        device_num = -1
        print("CUDA not available. Loading BART model on CPU.")

    classifier = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device_num
    )
    print("BART model loaded successfully.")
except Exception as e:
    print(f"Error loading BART model: {e}")