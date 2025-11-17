# 🎙️ ASVspoof Training and Evaluation Guide

### 📥 Step 1: Download the Datasets
- Download **ASVspoof 5**  
  Link: [https://zenodo.org/records/14498691](https://zenodo.org/records/14498691)  
  Files needed:  
  - `flac_T_aa.tar`  
  - `flac_T_bb.tar`

- Download **ASVspoof 2021 (for evaluation)**  
  Link: [https://zenodo.org/records/4837263](https://zenodo.org/records/4837263)

- Now, create another folder named asvspoof2019. e download ang folder nga naa ani dri sa gdrive: https://drive.google.com/drive/folders/1dtrVv2Z9V-k020tdSVYDiaGv6pV2Lg-7?usp=sharing
- Ibutang dayon sa folder na asvspoof2019 katong gi download sa gdrive.

### 📂 Step 2: Extract the Files
- After downloading, extract both datasets.  
- For **ASVspoof 5**, place the extracted files in a folder named **`asvspoof5`** (same directory as your source code).  
- For **ASVspoof 2021**, rename the extracted folder to **`asvspoof2021-eval`**.

**Example directory structure:**
```
Thesis training/
├── asvspoof5/
├── asvspoof2021-eval/
├── models/
├── protocols/
└── resnext.py
```

---

### 🎧 Step 3: Convert FLAC to WAV
- Open `convert_flac_to_wav.py`.
- Change the input folder path to the location of **asvspoof5**.
- Run the script to convert all `.flac` files to `.wav`.
- After finishing, do the same process for **asvspoof2021-eval**.

---

### 🧠 Step 4: Training
- Once all files are converted to `.wav`, you can start training the model.
- Make sure all dataset and protocol paths in the script are correct.
- Run the training file (e.g., `resnext.py`).

---

### 🧪 Step 5: Evaluation
- To evaluate, open your evaluation script.
- Change the **model path** to the trained model you want to test.
- Run the evaluation.
- The result will show the **Equal Error Rate (EER)** of your model.

---

 E TRAIN NA DAYON. 

---
