import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import re


def extract_features(df):
    """Добавляет числовые признаки к датафрейму"""
    df["len_words"] = df["esperanto"].apply(lambda x: len(x.split()))
    df["len_chars"] = df["esperanto"].apply(len)
    df["avg_word_len"] = df["len_chars"] / df["len_words"]
    df["num_punkt"] = df["esperanto"].apply(
        lambda x: len(re.findall(r"[!?]", x)))
    df["has_subjunctive"] = df["esperanto"].apply(
        lambda x: int(bool(re.search(r"\b(se|us)\b", x))))
    return df


def main():
    # Загрузка
    df = pd.read_csv("translated_cefr_dataset_merged.csv")
    df = df.dropna(subset=["esperanto", "cefr_level"])

    # Объединение A1 + A2 → A
    df["cefr_level"] = df["cefr_level"].replace({"A1": "A", "A2": "A"})

    # Извлечение признаков
    df = extract_features(df)

    # Векторизация текста
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    X_text = vectorizer.fit_transform(df["esperanto"])

    # Числовые признаки
    X_numeric = df[["len_words", "len_chars", "avg_word_len",
                    "num_punkt", "has_subjunctive"]].values
    X_combined = hstack([X_text, X_numeric])

    y = df["cefr_level"]

    # Сохранение
    joblib.dump(X_combined, "X_combined.joblib")
    joblib.dump(y, "y.joblib")
    joblib.dump(vectorizer, "cefr_vectorizer_best.joblib")

    print("✅ Признаки и вектор сохранены: X_combined.joblib, y.joblib, cefr_vectorizer_best.joblib")


if __name__ == "__main__":
    main()
