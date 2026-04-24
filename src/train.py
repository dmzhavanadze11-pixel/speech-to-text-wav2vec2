import torch
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from dataset import load_common_voice, preprocess_function
from collator import Wav2Vec2Collator


import os

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "data"))
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "models"))


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)
model.to(device)

# Load dataset
dataset = load_common_voice(DATA_PATH)

dataset = dataset.map(
    preprocess_function(processor),
    remove_columns=dataset["train"].column_names
)

dataset = dataset.filter(lambda x: x is not None)

# DataLoader
collator = Wav2Vec2Collator(processor)
train_loader = DataLoader(dataset["train"], batch_size=2, shuffle=True, collate_fn=collator)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

model.train()

for epoch in range(5):
    total_loss = 0

    for batch in train_loader:
        inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        labels = batch["labels"].to(device)

        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Save model
model.save_pretrained(MODEL_PATH)
processor.save_pretrained(MODEL_PATH)