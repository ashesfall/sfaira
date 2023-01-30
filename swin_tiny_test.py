from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import requests

extractor = AutoFeatureExtractor.from_pretrained("polejowska/swin-tiny-patch4-window7-224-eurosat")
model = AutoModelForImageClassification.from_pretrained("polejowska/swin-tiny-patch4-window7-224-eurosat")

image = Image.open("Tile.jpg").convert("RGB")

inputs = extractor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits

predicted_class_idx = logits.argmax(-1).item()
print("Predicted class: ", model.config.id2label[predicted_class_idx])