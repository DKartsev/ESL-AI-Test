# CEFR Text Classifier (A1–C2)

Классическая ML-модель (Logistic Regression), обученная на датасете [UniversalCEFR](https://huggingface.co/datasets/UniversalCEFR/cefr_sp_en), предсказывает уровень владения языком (A1–C2) по тексту. Используется TF-IDF и Gradio-интерфейс для ввода фраз. Планируется расширение под эсперанто через перевод.

## Установка

```bash
pip install -r requirements.txt
```

## Перевод датасета

Скрипт `translate_dataset.py` переводит 10&nbsp;000 английских примеров на эсперанто батчами по 16 предложений. На GPU процесс занимает около 5–10 минут, на CPU — до часа.

```bash
python translate_dataset.py
```

Результат сохраняется в `translated_cefr_dataset.csv`.

## Обучение модели

После перевода датасета запустите:

```bash
python train_model.py
```

Скрипт сохранит `cefr_model.joblib` и `cefr_vectorizer.joblib`.

## Запуск веб-интерфейса

```bash
python app.py
```

Откроется простое приложение на Gradio, в котором можно ввести фразу на эсперанто и получить прогнозируемый уровень.
