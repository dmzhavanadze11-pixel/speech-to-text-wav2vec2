import os
import pandas as pd
from datasets import Dataset
import torchaudio

def load_common_voice(data_folder):
    df = pd.read_csv(os.path.join(data_folder, "validated.tsv"), sep="\t")
    dataset = Dataset.from_pandas(df)

    def map_audio(batch):
        batch["audio"] = {
            "path": os.path.join(data_folder, "clips", batch["path"])
        }
        return batch

    dataset = dataset.map(map_audio)
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset


def preprocess_function(processor):
    def _fn(batch):
        speech, sr = torchaudio.load(batch["audio"]["path"])

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            speech = resampler(speech)

        batch["input_values"] = processor(
            speech.squeeze().numpy(),
            sampling_rate=16000
        ).input_values[0]

        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids

        return batch

    return _fn