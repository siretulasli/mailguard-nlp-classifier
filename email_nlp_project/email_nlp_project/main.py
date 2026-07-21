import os
import re
import string
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE


# ==============================
# SETTINGS
# ==============================

DATA_PATH = "final_email_dataset.csv"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==============================
# TEXT CLEANING
# ==============================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def save_class_distribution(y, title, filename):
    counts = y.value_counts()

    plt.figure(figsize=(9, 5))
    counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


def evaluate_model(model, model_name, experiment_name, X_train, X_test, y_train, y_test):
    print(f"\nTraining: {experiment_name} - {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    result = {
        "Experiment": experiment_name,
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

    safe_exp = experiment_name.replace(" ", "_").lower()
    safe_model = model_name.replace(" ", "_").lower()

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(
        os.path.join(OUTPUT_DIR, f"classification_report_{safe_exp}_{safe_model}.csv")
    )

    # Confusion matrix
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", values_format="d")
    plt.title(f"Confusion Matrix - {experiment_name} - {model_name}")
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_exp}_{safe_model}.png"),
        dpi=300
    )
    plt.close()

    print(result)
    return result


# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv(DATA_PATH)

print("Dataset loaded.")
print("Shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

df = df.dropna(subset=["text", "label"])
df["clean_text"] = df["text"].apply(clean_text)

save_class_distribution(
    df["label"],
    "Class Distribution Before SMOTE",
    "class_distribution_before_smote.png"
)


# ==============================
# TRAIN TEST SPLIT
# ==============================

X = df["clean_text"]
y = df["label"]

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train_text))
print("Test size:", len(X_test_text))


# ==============================
# TF-IDF
# ==============================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_test_tfidf = tfidf.transform(X_test_text)


# ==============================
# MODELS
# ==============================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Linear SVM": LinearSVC(
        random_state=42
    )
}


# ==============================
# EXPERIMENT 1: WITHOUT SMOTE
# ==============================

all_results = []

for model_name, model in models.items():
    result = evaluate_model(
        model=model,
        model_name=model_name,
        experiment_name="Without SMOTE",
        X_train=X_train_tfidf,
        X_test=X_test_tfidf,
        y_train=y_train,
        y_test=y_test
    )
    all_results.append(result)


# ==============================
# EXPERIMENT 2: WITH SMOTE
# ==============================

smote = SMOTE(random_state=42, k_neighbors=5)

X_train_smote, y_train_smote = smote.fit_resample(X_train_tfidf, y_train)

print("\nAfter SMOTE class distribution:")
print(pd.Series(y_train_smote).value_counts())

save_class_distribution(
    pd.Series(y_train_smote),
    "Class Distribution After SMOTE",
    "class_distribution_after_smote.png"
)

for model_name, model in models.items():
    result = evaluate_model(
        model=model,
        model_name=model_name,
        experiment_name="With SMOTE",
        X_train=X_train_smote,
        X_test=X_test_tfidf,
        y_train=y_train_smote,
        y_test=y_test
    )
    all_results.append(result)


# ==============================
# SAVE RESULT TABLES
# ==============================

results_df = pd.DataFrame(all_results)

results_df.to_csv(os.path.join(OUTPUT_DIR, "results_all.csv"), index=False)

results_df[results_df["Experiment"] == "Without SMOTE"].to_csv(
    os.path.join(OUTPUT_DIR, "results_without_smote.csv"),
    index=False
)

results_df[results_df["Experiment"] == "With SMOTE"].to_csv(
    os.path.join(OUTPUT_DIR, "results_with_smote.csv"),
    index=False
)

print("\nFinal Results:")
print(results_df)


# ==============================
# COMPARISON GRAPH
# ==============================

plot_df = results_df.copy()
plot_df["Label"] = plot_df["Experiment"] + "\n" + plot_df["Model"]

metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

x = range(len(plot_df))
bar_width = 0.18

plt.figure(figsize=(13, 7))

for i, metric in enumerate(metrics):
    positions = [p + (i - 1.5) * bar_width for p in x]
    plt.bar(positions, plot_df[metric], width=bar_width, label=metric)

plt.xticks(x, plot_df["Label"], rotation=20, ha="right")
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Model Performance Comparison With and Without SMOTE")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison_all_metrics.png"), dpi=300)
plt.close()


# ==============================
# BEST MODEL
# ==============================

best_row = results_df.sort_values(by="F1-score", ascending=False).iloc[0]

print("\nBest Model:")
print(best_row)

print("\nAll outputs saved in the outputs folder.")