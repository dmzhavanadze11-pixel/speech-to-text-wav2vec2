from dataclasses import dataclass
from typing import List, Dict, Union
import torch

@dataclass
class Wav2Vec2Collator:
    processor: any

    def __call__(self, features: List[Dict[str, Union[list, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_values = [f["input_values"] for f in features]
        labels = [f["labels"] for f in features]

        batch = self.processor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                {"input_ids": labels},
                padding=True,
                return_tensors="pt"
            )

        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100
        )

        return batch