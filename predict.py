import os
import csv
import torch
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

    torch.set_grad_enabled(False)
    model = util.load_model(model_name, labels)
    model.eval()
    print(f"Loaded Model: {model_name}")

    # mask = prediction_mask(io.imread("LU_labels_LIQ2018_CVc.tif"), numbers)
    # print(f"Prediction Mask Shape: {mask.shape}")

    imageData = io.imread("LC08_L1TP_044033_20180719_20200831_02_T1_B6_clip.TIF")
    print(f"Image Shape: {imageData.shape}")

    prediction = [[np.ndarray(shape=(len(labels)), dtype=np.float32) for j in range(imageData.shape[1])] for i in range(imageData.shape[0])]
    print(len(prediction))
    print(len(prediction[0]))

    granularity = 100
    size = 100

    total_rows = int(imageData.shape[0] / granularity) + 1
    print(f"Total Rows: {total_rows}")

    # Enumerate the entire image using a sliding window of 'size'
    # and stride of 'granularity;
    for x in range(0, imageData.shape[0], granularity):
        print(f"Predicting Row: {x / granularity} of {total_rows}")

        for y in range(0, imageData.shape[1], granularity):
            tileImage = util.extract_tile(imageData, x, y, size)
            io.imsave(f"eval/temp.jpeg", tileImage)
            value = {
                "image": [Image.open(f"eval/temp.jpeg")]
            }
            transforms(value)

            img_shape = value["pixel_values"][0].shape
            pixels = value["pixel_values"][0].reshape(1, img_shape[0], img_shape[1], img_shape[2])

            out = model.forward(pixel_values=pixels)

            # Apply this prediction to the entire window
            for i in range(x, x + granularity):
                for j in range(y, y + granularity):
                    if i < imageData.shape[0] and j < imageData.shape[1]:
                        prediction[i][j] = prediction[i][j] + out.logits[0].numpy()

    # Consolidate the predictions using argmax
    result = [[0 for j in range(imageData.shape[1])] for i in range(imageData.shape[0])]

    for x in range(0, imageData.shape[0]):
        print(f"Consolidating Row: {x} of {imageData.shape[0]}")

        for y in range(0, imageData.shape[1]):
            result[x][y] = numbers[np.argmax(prediction[x][y])]

    io.imsave("eval/result.tif", result)

main()