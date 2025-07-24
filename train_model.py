import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Загрузка данных
df = pd.read_csv("translated_cefr_dataset.csv")
X = df["text_eo"]
y = df["level"]

# Векторизация текста
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Разделение на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

# Обучение модели
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Оценка
y_pred = model.predict(X_test)
print("🔍 Классификационный отчёт:\n", classification_report(y_test, y_pred))

# Сохранение модели и векторизатора
joblib.dump(model, "cefr_model.joblib")
joblib.dump(vectorizer, "cefr_vectorizer.joblib")
print("✅ Модель и вектор сохранены.")
