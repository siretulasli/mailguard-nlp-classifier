# MailGuard NLP Classifier

MailGuard NLP Classifier is a multiclass machine learning project designed to identify different types of email-based cybersecurity threats.

The project uses natural language processing techniques to classify email messages into six categories and compares the performance of Logistic Regression and Linear Support Vector Machine models.

## Project Objective

Email attacks can involve more than traditional phishing. Social engineering messages may also contain malware, scareware, baiting, or pretexting techniques.

The objective of this project is to create an NLP-based classification pipeline that can distinguish legitimate emails from multiple malicious email categories.

The supported classes are:

- Phishing
- Malware
- Scareware
- Baiting
- Pretexting
- Non-malicious

## Dataset

The dataset contains 621 labeled email samples distributed across six categories.

| Class | Samples |
|---|---:|
| Non-malicious | 171 |
| Phishing | 114 |
| Scareware | 100 |
| Baiting | 80 |
| Malware | 78 |
| Pretexting | 78 |
| **Total** | **621** |

The original dataset is stored as an Excel file and converted into a clean CSV format using `prepare_dataset.py`.

The prepared dataset contains two main columns:

```text
text
label
```

## Project Workflow

```text
Raw Email Dataset
        ↓
Dataset Preparation
        ↓
Text Cleaning
        ↓
Train-Test Split
        ↓
TF-IDF Feature Extraction
        ↓
Model Training
        ↓
SMOTE Class Balancing
        ↓
Model Evaluation
```

## Text Preprocessing

The preprocessing stage includes:

- Converting text to lowercase
- Removing URLs
- Removing punctuation
- Removing numeric characters
- Removing English stop words
- Removing missing or empty records

## Feature Extraction

Email messages are converted into numerical features using TF-IDF.

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
```

The model therefore considers both individual words and two-word combinations.

## Machine Learning Models

The project compares two supervised machine learning algorithms.

### Logistic Regression

Logistic Regression is used as a strong baseline model for multiclass text classification.

### Linear Support Vector Machine

Linear SVM is suitable for high-dimensional and sparse feature spaces produced by TF-IDF.

## Experimental Setup

Four different model configurations are evaluated:

1. Logistic Regression without SMOTE
2. Linear SVM without SMOTE
3. Logistic Regression with SMOTE
4. Linear SVM with SMOTE

The dataset is divided using a stratified train-test split:

```text
Training set: 80%
Test set: 20%
Random state: 42
```

SMOTE is applied only to the training data to reduce class imbalance without affecting the test set.

## Results

| Experiment | Model | Accuracy | Precision | Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| Without SMOTE | Logistic Regression | 89.60% | 93.73% | 87.62% | 89.83% |
| Without SMOTE | Linear SVM | 91.20% | 91.73% | 90.13% | 90.76% |
| With SMOTE | Logistic Regression | **92.00%** | 92.79% | **91.17%** | **91.80%** |
| With SMOTE | Linear SVM | **92.00%** | **92.95%** | 90.85% | 91.70% |

The highest macro F1-score was achieved by Logistic Regression with SMOTE, reaching approximately 91.8%.

Both SMOTE-based models achieved 92% test accuracy.

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Classification report
- Confusion matrix

Macro-averaged metrics are used because they assign equal importance to every email category.

## Technologies

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Imbalanced-learn
- OpenPyXL
- TF-IDF
- SMOTE
- Logistic Regression
- Linear SVM

## Project Structure

```text
mailguard-nlp-classifier/
│
├── main.py
├── prepare_dataset.py
├── final_email_dataset.csv
├── phishing_nlp_dataset.xlsx
├── requirements.txt
├── README.md
│
└── outputs/
    ├── results_all.csv
    ├── results_without_smote.csv
    ├── results_with_smote.csv
    └── classification_reports/
```

## Installation

Clone the repository:

```bash
git clone https://github.com/siretulasli/mailguard-nlp-classifier.git
```

Move into the project directory:

```bash
cd mailguard-nlp-classifier
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

The `requirements.txt` file should contain:

```text
pandas
matplotlib
scikit-learn
imbalanced-learn
openpyxl
```

## Usage

Run the complete model training and evaluation pipeline:

```bash
python main.py
```

The program will:

- Load the prepared email dataset
- Clean the email text
- Generate TF-IDF features
- Train four model configurations
- Apply SMOTE to the training data
- Evaluate each model
- Print classification reports
- Save evaluation results

To recreate the CSV dataset from the original Excel file:

```bash
python prepare_dataset.py
```

Then run:

```bash
python main.py
```

## Key Findings

The experiments show that:

- TF-IDF features are effective for email-threat classification.
- Linear SVM performs strongly without oversampling.
- SMOTE improves the balance between different email classes.
- Logistic Regression with SMOTE achieves the highest macro F1-score.
- The system can distinguish multiple social engineering attacks instead of performing only binary phishing detection.

## Limitations

- The dataset is relatively small.
- The project currently supports English email text.
- TF-IDF does not fully understand semantic context.
- The system is currently an offline machine learning experiment.
- Predictions should not be used as the only security control in a production environment.

## Future Improvements

- Use a larger real-world email dataset
- Add cross-validation and hyperparameter tuning
- Add transformer-based language models
- Build a real-time email classification interface
- Add probability or confidence scores
- Create an API for model predictions
- Add automated tests
- Save and load trained models with Joblib

## Security Notice

This project is developed for educational and research purposes.

Do not upload private emails, credentials, API keys, passwords, or sensitive company information to a public repository.

## License

This project is licensed under the MIT License.

## Disclaimer

The classification results are experimental and should not replace professional cybersecurity tools, secure email gateways, or manual security analysis.
