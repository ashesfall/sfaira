import csv
from skimage import io
from PIL import Image

import numpy as np
from datasets import Dataset
import evaluate
from transformers import AutoModelForImageClassification, TrainingArguments, DefaultDataCollator

data_collator = DefaultDataCollator()
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    evaluation_strategy="steps",
    num_train_epochs=12,
    fp16=False,
    save_steps=100,
    eval_steps=100,
    logging_steps=10,
    learning_rate=2e-4,
    save_total_limit=2,
    remove_unused_columns=False,
)

def load_model(model_name, labels):
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    return model

def prepare_input(imageData, granularity, size, transforms):
    count = 0

    tiles = []

    for x in range(0, imageData.shape[0], granularity):
        for y in range(0, imageData.shape[1], granularity):
            tileImage = extract_tile(imageData, x, y, size)
            io.imsave(f"eval/{count}.jpeg", tileImage)
            value = {
                "image": [Image.open(f"eval/{count}.jpeg")]
            }
            transforms(value)
            tiles.append(value["pixel_values"][0])
            count = count + 1

    return tiles

def create_dataset(labelData, imageData, size, threshold, validationRatio=0):
    threshold = size * size * threshold
    count = 0

    tiles = []

    for x in range(0, labelData.shape[0], size):
        for y in range(0, labelData.shape[1], size):
            tileData = extract_tile(labelData, x, y, size)
            l = label(tileData, threshold)

            validation = False
            if validationRatio > 0:
                validation = count % validationRatio == 0;

            if l > 0:
                tileImage = extract_tile(imageData, x, y, size)
                if validation:
                    io.imsave(f"validation/{count}.jpeg", tileImage)
                else:
                    io.imsave(f"train/{count}.jpeg", tileImage)
                    tiles.append({
                        "image": Image.open(f"train/{count}.jpeg"),
                        "label": l
                    })
                count = count + 1

    print(f"Total tiles: {count}")
    dataset = Dataset.from_list(tiles)
    return dataset

def load_labels(file):
    numbers = []
    labels = []

    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        first = True
        for row in reader:
            if first:
                first = False
                continue
            numbers.append(int(row[0]))
            labels.append(row[1])

    return labels, numbers

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

metric = evaluate.load("accuracy")
def compute_metrics(p):
    return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)
