import csv
from skimage import io
from PIL import Image

import numpy as np
from datasets import Dataset
from transformers import Trainer
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, TrainingArguments, DefaultDataCollator

from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor

model_name = "google/vit-base-patch16-224-in21k"
# model_name = "polejowska/swin-tiny-patch4-window7-224-eurosat"

extractor = AutoFeatureExtractor.from_pretrained(model_name)
normalize = Normalize(mean=extractor.image_mean, std=extractor.image_std)
_transforms = Compose([RandomResizedCrop(extractor.size), ToTensor(), normalize])

def main():
    labelData = io.imread("LU_labels_LIQ2018_CVc.tif")
    imageData = io.imread("LC08_L1TP_044033_20180719_20200831_02_T1_B6_clip.TIF")

    print(f"Label Shape: {labelData.shape}")
    print(f"Image Shape: {imageData.shape}")

    labels = load_labels("C2VSimLanduseCategories.csv")
    print(f"Labels: {labels}")
    dataset = create_dataset(labelData, imageData, 100, 0.5)
    dataset = dataset.train_test_split(test_size=0.2)

    train(dataset.with_transform(transforms), labels)

def train(dataset, labels):
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    data_collator = DefaultDataCollator()
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=4,
        fp16=False,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        remove_unused_columns=False,
    )

    print("Creating Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=extractor,
    )

    trainer.train()

def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def create_dataset(labelData, imageData, size, threshold):
    threshold = size * size * threshold
    count = 0

    tiles = []

    for x in range(0, labelData.shape[0], size):
        for y in range(0, labelData.shape[1], size):
            tileData = extract_tile(labelData, x, y, size)
            l = label(tileData, threshold)
            if l > 0:
                tileImage = extract_tile(imageData, x, y, size)
                io.imsave(f"tiles/{count}.jpeg", tileImage)
                tiles.append({
                    "image": Image.open(f"tiles/{count}.jpeg"),
                    "label": l
                })
                count = count + 1

    print(f"Total tiles: {count}")
    dataset = Dataset.from_list(tiles)
    return dataset

def load_labels(file):
    labels = []

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        first = True
        for row in reader:
            if first:
                first = False
                continue
            labels.append(row[1])

    return labels

def label(tileData, threshold):
    dist = label_distribution(tileData)
    label = max(dist, key=dist.get)
    if (dist[label] < threshold) & (label < 27):
        return -1
    return label - 1

def label_distribution(tileData):
    unique, counts = np.unique(tileData, return_counts=True)
    return dict(zip(unique, counts))

def extract_tile(data, x, y, size):
    return data[x:x+size, y:y+size]

main()