import gradio as gr
import joblib

# Загрузка модели и векторизатора
model = joblib.load("cefr_model.joblib")
vectorizer = joblib.load("cefr_vectorizer.joblib")

# Предсказание уровня
def predict_level(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return f"🌍 Прогнозируемый уровень CEFR: **{prediction}**"

# Интерфейс
iface = gr.Interface(
    fn=predict_level,
    inputs=gr.Textbox(lines=2, placeholder="Введите фразу на эсперанто..."),
    outputs="markdown",
    title="📊 Оценка уровня владения языком (CEFR)",
    description="Введите фразу на языке **эсперанто**, чтобы получить примерный уровень владения по шкале CEFR (A1–C2)."
)

if __name__ == "__main__":
    iface.launch()
