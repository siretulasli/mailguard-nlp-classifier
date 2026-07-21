Project Title:
Multiclass Email Classification Using NLP and Machine Learning

Students:
Ali Toprak Tuğtekin - 220601073
Efe Bora - 220601046

Description:
This project performs multiclass email classification using Natural Language Processing and machine learning techniques. The emails are classified into six categories: phishing, malware, scareware, baiting, pretexting, and non-malicious.

Files:
- prepare_dataset.py: Converts the original Excel dataset into the final CSV dataset.
- main.py: Trains and evaluates Logistic Regression and Linear SVM models with and without SMOTE.
- final_email_dataset.csv: Final cleaned dataset used in the experiments.
- Outputs folder: Contains result tables, classification reports, confusion matrices, and performance graphs.

How to Run:
1. Install the required libraries:
   pip install -r requirements.txt

2. Prepare the dataset:
   python prepare_dataset.py

3. Train and evaluate the models:
   python main.py

Methods:
- Text preprocessing
- TF-IDF feature extraction
- Logistic Regression
- Linear SVM
- SMOTE for class imbalance handling
- Accuracy, Precision, Recall, and F1-score evaluation

Best Result:
Logistic Regression with SMOTE achieved the best F1-score of 0.918.