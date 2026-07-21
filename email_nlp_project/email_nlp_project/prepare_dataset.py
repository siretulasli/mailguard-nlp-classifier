import pandas as pd

df = pd.read_excel("phishing_nlp_dataset.xlsx")

df["label"] = df["Corpus"].str.extract(r"\t([A-Za-z\- ]+)$")[0]
df["text"] = df["Corpus"].str.replace(r"\t([A-Za-z\- ]+)$", "", regex=True)

df = df.dropna(subset=["text", "label"])

label_map = {
    "NOT-Malicious General Class": "non-malicious",
    "Phishing": "phishing",
    "Scareware": "scareware",
    "Baiting": "baiting",
    "Malware": "malware",
    "Pretexting": "pretexting"
}

df["label"] = df["label"].map(label_map)
df = df.dropna(subset=["label"])

df = df[["text", "label"]]
df.to_csv("final_email_dataset.csv", index=False)

print("Final dataset oluşturuldu.")
print("Dataset boyutu:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())