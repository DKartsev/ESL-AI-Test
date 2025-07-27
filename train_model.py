import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def main():
    # Загрузка признаков и целевой переменной
    X = joblib.load("X_combined.joblib")
    y = joblib.load("y.joblib")

    print("📊 Распределение классов:")
    print(pd.Series(y).value_counts())

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Параметры для поиска
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["saga"],
        "class_weight": [None, "balanced"],
    }

    # Поиск по сетке
    grid = GridSearchCV(
        LogisticRegression(max_iter=2000),
        param_grid,
        cv=5,
        scoring="accuracy",
        verbose=1,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    # Сохранение результатов
    results_df = pd.DataFrame(grid.cv_results_).sort_values(
        by="mean_test_score", ascending=False)
    results_df.to_csv("grid_search_results_extended.csv", index=False)
    print("📄 Результаты GridSearchCV сохранены в grid_search_results_extended.csv")

    print(f"✅ Лучшая модель: {grid.best_estimator_}")
    print(f"📈 Лучшая точность: {grid.best_score_:.4f}")

    # Оценка на тесте
    y_pred = grid.predict(X_test)
    print("🔍 Отчёт по качеству классификации:")
    print(classification_report(y_test, y_pred))

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=sorted(
        set(y)), yticklabels=sorted(set(y)), cmap="Blues")
    plt.xlabel("Предсказано")
    plt.ylabel("Истинное значение")
    plt.title("📉 Матрица ошибок (расширенные признаки)")
    plt.tight_layout()
    plt.savefig("confusion_matrix_extended.png")
    plt.close()

    # Сохранение модели
    joblib.dump(grid.best_estimator_, "cefr_model_extended.joblib")
    print("✅ Модель сохранена: cefr_model_extended.joblib")


if __name__ == "__main__":
    main()
