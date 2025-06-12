from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import evaluate
from datasets import load_dataset
from tqdm import tqdm
import torch


# Initialize CER metric
cer_metric = evaluate.load("cer")

# processor = TrOCRProcessor.from_pretrained('microsoft/trocr-small-handwritten', cache_dir='./model')
# model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-small-handwritten', cache_dir='./model')
processor = TrOCRProcessor.from_pretrained('./model/save')
model = VisionEncoderDecoderModel.from_pretrained('./model/save')
model.to("cuda:0").eval()

dataset = load_dataset("parquet", data_files={
        'test': './data/data/test.parquet'
    })["test"]


def batch_predict(images):
    """
    Performs batch OCR prediction.

    Args:
        images (list of PIL.Image.Image): A list of PIL images.

    Returns:
        list of str: A list of predicted text strings.
    """
    pixel_values = processor(images=images, return_tensors="pt").pixel_values.to("cuda:0")
    generated_ids = model.generate(pixel_values)
    predicted_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return predicted_texts



def calculate_cer_batch(examples, batch_size=16):
    """
    Performs batch OCR prediction and calculates the Character Error Rate (CER).

    Args:
        examples (datasets.Dataset): The input dataset with 'image' and 'text' columns.
        batch_size (int): The size of the batch for processing.  Adjust based on GPU memory.

    Returns:
        dict: A dictionary containing lists of predicted text and CER values.
    """
    predicted_texts = []
    cers = []
    images = []
    ground_truth_texts = []

    for i in tqdm(range(len(examples))):
      images.append(examples[i]["image"].convert("RGB"))
      ground_truth_texts.append(examples[i]["text"])

    for i in tqdm(range(0, len(images), batch_size)):
        batch_images = images[i:i + batch_size]
        batch_ground_truth_texts = ground_truth_texts[i:i + batch_size]

        predicted_texts_batch = batch_predict(batch_images)
        predicted_texts.extend(predicted_texts_batch)

        cers_batch = cer_metric.compute(predictions=predicted_texts_batch, references=batch_ground_truth_texts)
        if isinstance(cers_batch, float):
            # Single batch edge case
            cers.extend([cers_batch] * len(predicted_texts_batch))
        elif isinstance(cers_batch, list):
            #Multiple batches
            cers.extend(cers_batch)  # Ensure correct number of cer values for each prediction
        else:
            raise TypeError(f"Unexpected type for CER: {type(cers_batch)}")


    return {"predicted_text": predicted_texts, "cer": cers}


# Process the dataset in batches and calculate CER
results = calculate_cer_batch(dataset)


# Calculate the average CER over the entire dataset
average_cer = sum(results["cer"]) / len(dataset)

print(f"Average CER on the test set: {average_cer}")