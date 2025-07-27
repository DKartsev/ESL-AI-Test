import gradio as gr
import joblib

# Пути к модели и векторизатору
model_path = "models/cefr_model_best.joblib"
vectorizer_path = "models/cefr_vectorizer_best.joblib"

# Загрузка модели и векторизатора
model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

# Предсказание
def predict_cefr(text):
    if not text.strip():
        return "⛔ Введите предложение на эсперанто."
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    return f"🔤 Предсказанный уровень: {prediction}"

# Интерфейс
interface = gr.Interface(
    fn=predict_cefr,
    inputs=gr.Textbox(lines=2, placeholder="Введи фразу на эсперанто..."),
    outputs="text",
    title="📘 Оценка уровня CEFR на эсперанто",
    description="Введи предложение на языке эсперанто, и модель оценит его уровень по шкале A1–C2.",
    examples=[
        ["Mi manĝas pomon."],
        ["Ŝi legis interesan romanon."],
        ["Ni diskutis la filozofiajn ideojn."],
        ["La intertempa kunteksto kunfandiĝis kun sintaksa komplekseco."],
    ]
)

if __name__ == "__main__":
    interface.launch()
