import pandas as pd
import re


def is_suspicious_pair(eo: str, en: str) -> bool:
    # 1. Длина предложения
    eo_len = len(eo.split())
    en_len = len(en.split())
    if eo_len < 3 or eo_len > 25:
        return True
    if eo_len > en_len * 2 or en_len > eo_len * 2:
        return True

    # 2. Повторы символов или слогов
    if re.search(r"\b(\w+)\1+\b", eo):  # повтор слов
        return True
    if re.search(r"(..+?)\1{2,}", eo):  # повтор слогов
        return True
    if re.search(r"([a-zĉĝĥĵŝŭ])\1{3,}", eo):  # повтор символа ≥ 4
        return True

    # 3. Странные токены
    if re.search(r"[\"']s\b", eo):  # англ possessive
        return True
    if re.search(r"[^a-zA-ZĉĝĥĵŝŭĈĜĤĴŜŬ ,.\-?!0-9]", eo):  # странные символы
        return True

    return False


def main():
    df = pd.read_csv("translated_cefr_dataset.csv")
    df["is_suspicious"] = df.apply(lambda row: is_suspicious_pair(
        str(row["esperanto"]), str(row["english"])), axis=1)

    # Сохраняем новый файл
    df.to_csv("translated_cefr_dataset_filtered.csv", index=False)
    print(
        f"✅ Обнаружено подозрительных фраз: {df['is_suspicious'].sum()} / {len(df)}")


if __name__ == "__main__":
    main()
