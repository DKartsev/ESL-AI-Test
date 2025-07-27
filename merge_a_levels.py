# merge_a_levels.py

import pandas as pd


def main():
    input_file = "translated_cefr_dataset_clean_combined.csv"
    output_file = "translated_cefr_dataset_merged.csv"

    # Загрузка данных
    df = pd.read_csv(input_file)

    # Проверка, что нужные столбцы есть
    if "cefr_level" not in df.columns:
        raise ValueError("В файле не найден столбец 'cefr_level'.")

    # Объединение уровней A1 и A2 → A
    df["cefr_level"] = df["cefr_level"].replace({"A1": "A", "A2": "A"})

    # Сохранение
    df.to_csv(output_file, index=False)
    print(f"✅ Объединённый датасет сохранён как {output_file}")

    # Статистика
    print("\n📊 Новое распределение уровней:")
    print(df["cefr_level"].value_counts())


if __name__ == "__main__":
    main()
