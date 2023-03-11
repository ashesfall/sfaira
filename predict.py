import os
import csv
from skimage import io
from PIL import Image

import util
import numpy as np
from datasets import Dataset
from transformers import Trainer
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, DefaultDataCollator

from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

model_name = "./results/checkpoint-600"

extractor = AutoFeatureExtractor.from_pretrained(model_name)
normalize = Normalize(mean=extractor.image_mean, std=extractor.image_std)
_transforms = Compose([RandomResizedCrop(extractor.size), ToTensor(), normalize])

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def predict(dataset, labels):
    model = util.load_model(model_name, labels)

    trainer = Trainer(
        model=model,
        args=util.training_args,
        data_collator=util.data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=util.compute_metrics,
        tokenizer=extractor,
    )

    metrics = trainer.evaluate(dataset['test'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

def prediction_mask(labelExample, labels):
    mask = np.zeros(labelExample.shape)

    for i in range(len(labelExample)):
        for j in range(len(labelExample[i])):
            if labelExample[i][j] in labels:
                mask[i][j] = 1
            else:
                mask[i][j] = 0

    return mask

def validation():
    labels, numbers = util.load_labels("C2VSimLanduseCategories.csv")
    print(f"Labels: {labels}")

    labelData = io.imread("LU_labels_LIQ2018_CVc.tif")
    print(f"Label Data Shape: {labelData.shape}")

    imageData = io.imread("LC08_L1TP_044033_20180719_20200831_02_T1_B6_clip.TIF")
    print(f"Image Shape: {imageData.shape}")

    dataset = util.create_dataset(labelData, imageData, 100, 0.5)
    dataset = dataset.train_test_split(test_size=0.5)
    predict(dataset.with_transform(transforms), labels)

def main():
    labels, numbers = util.load_labels("C2VSimLanduseCategories.csv")
    print(f"Labels: {labels}")

    model = util.load_model(model_name, labels)
    model.eval()
    print(f"Loaded Model: {model_name}")

    # mask = prediction_mask(io.imread("LU_labels_LIQ2018_CVc.tif"), numbers)
    # print(f"Prediction Mask Shape: {mask.shape}")

    imageData = io.imread("LC08_L1TP_044033_20180719_20200831_02_T1_B6_clip.TIF")
    print(f"Image Shape: {imageData.shape}")

    prediction = np.ndarray(shape=(imageData.shape[0], imageData.shape[1], len(labels)), dtype=np.float32)
    count = np.ndarray(shape=(imageData.shape[0], imageData.shape[1]), dtype=np.float32)

    size = 100

    input = np.array(util.prepare_input(imageData, 100, transforms))
    print(input.shape)
    print(f"Model.forward: {model.forward(pixel_values=input, labels=labels)}")

main()