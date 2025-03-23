from flask import Flask, render_template, request
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)

# Load pre-trained model and tokenizer
MODEL_NAME = "models/student-model-lora"
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
labels = ["Non-Toxic", "Toxic"]

def classify_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    return labels[prediction]

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        user_text = request.form["text_input"]
        result = classify_text(user_text)
    return render_template("home.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
