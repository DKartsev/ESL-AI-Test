# CEFR Text Classifier (A1–C2)

Классическая ML-модель (Logistic Regression), обученная на датасете [UniversalCEFR](https://huggingface.co/datasets/UniversalCEFR/cefr_sp_en), предсказывает уровень владения языком (A1–C2) по тексту. Используется TF-IDF и Gradio-интерфейс для ввода фраз. Планируется расширение под эсперанто через перевод.

Запуск: `app.py`  
Тренировка: `train_model.py`
