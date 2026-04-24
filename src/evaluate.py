import torch
import jiwer
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

from dataset import load_common_voice, preprocess_function
import os


DATA_PATH = "C:\\Users\\user\\Desktop\\SP2TXT\\data"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_PATH = os.path.abspath(MODEL_PATH)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
model.eval()

dataset = load_common_voice(DATA_PATH)
dataset = dataset.map(preprocess_function(processor))

predictions = []
references = []

for sample in dataset["test"]:
    input_values = processor(
        sample["input_values"],
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred = processor.decode(pred_ids[0])

    predictions.append(pred)
    references.append(sample["sentence"])

wer = jiwer.wer(references, predictions)
print("WER:", wer)