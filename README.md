# Fraud-Detection-Pipeline


This repository implements a simple end-to-end fraud detection workflow: data loading, class balancing, model training, and evaluation.

## 📂 Repository Structure

```
.
├── fraud_detection_pipeline.py    # Main script
├── data/
│   └── transactions.csv           # Raw input data (example)
├── notebooks/
│   └── EDA_and_visualization.ipynb  # (Optional) exploratory analysis
├── outputs/
│   ├── confusion_matrix.png
│   └── roc_curve.png
├── .env                           # (not tracked) API keys or secrets
├── .gitignore
├── requirements.txt
└── README.md
```

## 🔧 Prerequisites

- Python 3.8+  
- pip (or conda)

## 📥 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/fraud-detection-pipeline.git
   cd fraud-detection-pipeline
   ```

2. **Create & activate a virtual environment**  
   ```bash
   python -m venv venv
   source venv/bin/activate       # macOS/Linux
   venv\Scripts\activate        # Windows
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ Configuration

1. **.env file**  
   Create a file named `.env` in the project root with any secrets or paths you need.  
   ```env
   DATA_PATH=./data/transactions.csv
   ```

2. **.gitignore**  
   Ensure your `.gitignore` contains at least:  
   ```
   venv/
   __pycache__/
   .env
   outputs/
   ```

## 🚀 Usage

Run the pipeline script end-to-end:

```bash
python fraud_detection_pipeline.py   --data-path "$DATA_PATH"   --output-dir ./outputs   --test-size 0.2   --random-state 42
```

| Option           | Description                                                       | Default   |
| ---------------- | ----------------------------------------------------------------- | --------- |
| `--data-path`    | Path to the CSV file containing raw transactions                  | `./data/transactions.csv` |
| `--output-dir`   | Directory to save plots and metrics                               | `./outputs` |
| `--test-size`    | Fraction of data held out for testing                             | `0.2`     |
| `--random-state` | Seed for reproducibility                                          | `42`      |

After running, you’ll find:

- **`confusion_matrix.png`**  
- **`roc_curve.png`**  

in the `./outputs` folder.

## 📈 What’s Inside

- **Data Loading & Cleaning**  
- **Train/Test Split** with stratification  
- **Class Balancing**: SMOTE & NearMiss  
- **Model Training**: RandomForest & LogisticRegression  
- **Evaluation**: F1 score, ROC curve, confusion matrix  

## 🛠️ Extending

- Swap in a different classifier by editing the `models` list.  
- Add hyperparameter tuning with GridSearchCV.  
- Turn script into a Python module or package for import.
