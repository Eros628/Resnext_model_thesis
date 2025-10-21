# Lightweight Deepfake Audio Detection via ResNet to MobileNet Knowledge Distillation

This repository contains the implementation for a deepfake audio detection system using knowledge distillation, trained on the ASVspoof datasets.

# Project Structure

The project is organized into a modular structure for clarity and scalability:

```
deepfake-audio-detection/
├── data/
│   ├── raw/
│   └── protocols/
├── saved_models/
├── scripts/
│   └── prepare_data.py
├── src/
│   ├── data_loader.py
│   ├── models/
│   ├── trainer.py
│   └── utils.py
├── config.py
├── evaluate.py
├── requirements.txt
└── train.py
```

# Getting Started

Follow these steps to set up the project, train the model, and evaluate its performance.

## Step 1: Setup and Installation

### 1. Clone the repository:

```
git clone <your-repository-url>
cd deepfake-audio-detection
```

### 2. Install dependencies: It's highly recommended to use a virtual environment.

```
pip install -r requirements.txt
```

## Step 2: Data Preparation

This step involves downloading and organizing all the necessary datasets.

### 1. Download the Audio Data:

- ASVspoof 2019/2021: Download the audio files for ASVspoof 2019 (train/dev) and 2021 (eval).
- ASVspoof 5: Download the audio files from the [Zenodo Link](https://zenodo.org/records/14498691).

### 2. Download Protocol Files:

- Download the ASVspoof 2019 protocol/metadata files from this [Google Drive link](https://drive.google.com/drive/folders/1dtrVv2Z9V-k020tdSVYDiaGv6pV2Lg-7?).

### 3. Organize the Folders:

- Extract and place all downloaded files into the data/ directory. Your final structure should look like this:

```
data/
├── raw/
│   ├── asvspoof2019_train/      # Contains 2019 training audio
│   ├── asvspoof2019_dev/        # Contains 2019 development audio
│   ├── asvspoof2021_eval/       # Contains 2021 evaluation audio
│   └── asvspoof5/               # Contains ASVspoof 5 audio
│
└── protocols/
    └── ASVspoof2019.LA.cm/      # Contains .txt files from Google Drive
```

## Step 3: Preprocessing (FLAC to WAV)

Run the provided script to convert all .flac audio files to the .wav format required for training.

```
python scripts/prepare_data.py
```

## Step 4: Configuration

Before running the training or evaluation, verify all paths and hyperparameters in the central configuration file.

- Open config.py and ensure that all directory paths (e.g., TRAIN_AUDIO_DIR, PROTOCOLS_DIR) match the locations on your machine.

## Step 5: Model Training

Execute the main training script to start the training process. The script will handle data loading, model training, and saving the best model checkpoint.

```
python train.py
```

- Progress will be displayed in the console.

- The best-performing model will be saved automatically in the saved_models/ directory.

## Step 6: Model Evaluation

To evaluate your trained model on the ASVspoof 2021 evaluation set:

- Update Config: Open config.py and make sure the EVAL_MODEL_PATH variable points to your saved model (e.g., saved_models/resnext_teacher_best.pth).
- Run Evaluation:

```
python evaluate.py
```
