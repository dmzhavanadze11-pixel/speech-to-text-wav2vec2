# speech-to-text-wav2vec2# 🎤 Georgian Speech-to-Text (Wav2Vec2)

A deep learning speech recognition system for the Georgian language using **HuggingFace Wav2Vec2**.

---

## 🚀 Project Overview

This project fine-tunes a **Wav2Vec2ForCTC** model to transcribe Georgian speech into text using the Mozilla Common Voice dataset.

It includes:

* Training pipeline
* Evaluation (WER metric)
* Real-time microphone transcription
* Custom dataset preprocessing

---

## 🧠 Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* Torchaudio
* Datasets (Common Voice)

---

## 📁 Project Structure

```text id="proj1"
SP2TXT/
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── realtime.py
│   ├── dataset.py
│   ├── collator.py
├── models/        (ignored in git)
├── data/          (ignored in git)
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash id="inst1"
pip install -r requirements.txt
```

If no requirements file:

```bash id="inst2"
pip install torch torchaudio transformers datasets jiwer sounddevice pandas
```

---

## ▶️ Usage

### 1. Train model

```bash id="train1"
python src/train.py
```

---

### 2. Evaluate model (WER)

```bash id="eval1"
python src/evaluate.py
```

---

### 3. Real-time transcription

```bash id="rt1"
python src/realtime.py
```

---

## 📊 Example Output

```text id="ex1"
Input:  გამარჯობა
Output: გამარჯობა
WER: 0.18
```

---

## ⚠️ Important Notes

* Dataset (`data/`) is not included (too large for GitHub)
* Model weights (`models/`) are excluded
* You can train the model locally using `train.py`

---

## 📌 Dataset

* Mozilla Common Voice (Georgian)

---

## 👨‍💻 Author

**Davit Mzhavanadze**

---

## ⭐ Future Improvements

* Add web UI (Flask / FastAPI)
* Deploy model API
* Improve WER with more data
* Export model to HuggingFace Hub
