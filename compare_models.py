import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.utils import shuffle

# Инженерия признаков: длина текста, число слов и пр.


def extract_text_features(texts):
    return np.array([
        [len(t), len(t.split()), sum(c.isupper() for c in t), sum(
            c.isdigit() for c in t), sum(c in "?!.,;" for c in t)]
        for t in texts
    ])


class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return extract_text_features(X)


# Загружаем данные
df = pd.read_csv("translated_cefr_dataset_clean_combined.csv").dropna(
    subset=["esperanto", "cefr_level"])
df["cefr_level"] = df["cefr_level"].replace({"A1": "A", "A2": "A"})
df = shuffle(df, random_state=42)

X = df["esperanto"]
y = df["cefr_level"]

# Общий пайплайн с объединением признаков


def make_pipeline(clf):
    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("stats", TextStats())
        ])),
        ("clf", clf)
    ])


# Кандидаты
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LinearSVC": LinearSVC(),
    "MultinomialNB": MultinomialNB(),
    "GradientBoosting": GradientBoostingClassifier()
}

# Сравнение
print("📊 Сравнение моделей (точность по 5-fold CV):\n")
for name, model in models.items():
    pipeline = make_pipeline(model)
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"{name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
