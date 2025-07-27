import pandas as pd

df = pd.read_csv("translated_cefr_dataset_filtered.csv")
df["cefr_level"] = df["cefr_level"].replace({"C1": "C", "C2": "C"})
df_clean = df[df["is_suspicious"] == False].reset_index(drop=True)
df_clean.to_csv("translated_cefr_dataset_clean_combined.csv", index=False)
print("✅ Датасет сохранён.")
