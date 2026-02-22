
import streamlit as st
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
import numpy as np

# ---------------------------
# Load Model Architecture
# ---------------------------
class DeepSignalModel(nn.Module):
    def __init__(self):
        super(DeepSignalModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, 3)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


# ---------------------------
# Load Saved Model
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from huggingface_hub import hf_hub_download

MODEL_REPO = "SSHHIVANI/deepsignal-ai-safety-engine"

# Download model weights from Hugging Face
model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename="deepsignal_model.pt"
)

model = DeepSignalModel()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Load tokenizer directly from Hugging Face
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_REPO)


# ---------------------------
# Risk Functions
# ---------------------------
def compute_risk_score(probs):
    return (
        0.5 * probs[0] +
        0.3 * probs[1] +
        0.2 * probs[2]
    )

def categorize_risk_advanced(probs):
    depression, toxicity, manipulation = probs
    
    if depression > 0.9:
        return "Critical Risk"
    
    if toxicity > 0.8:
        return "High Toxicity Risk"
    
    if manipulation > 0.9:
        return "High Manipulation Risk"
    
    score = compute_risk_score(probs)
    
    if score < 0.2:
        return "Safe"
    elif score < 0.4:
        return "Monitor"
    elif score < 0.7:
        return "Moderate Risk"
    else:
        return "Critical Risk"


# ---------------------------
# Prediction Function
# ---------------------------
def predict_text(text):
    encoding = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]
    
    thresholds = [0.45, 0.2, 0.1]
    preds = (probs > thresholds).astype(int)
    
    risk_score = compute_risk_score(probs)
    risk_category = categorize_risk_advanced(probs)
    
    return probs, preds, risk_score, risk_category


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="DeepSignal AI Safety Engine", layout="centered")

st.title("🧠 DeepSignal AI Safety Intelligence")
st.markdown("Multi-Label Mental Health, Toxicity & Manipulation Detection")

user_input = st.text_area("Enter text to analyze:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        probs, preds, risk_score, risk_category = predict_text(user_input)

        st.subheader("Prediction Results")

        st.write("### Probabilities")
        st.write({
            "Depression": float(probs[0]),
            "Toxicity": float(probs[1]),
            "Manipulation": float(probs[2])
        })

        st.write("### Risk Score")
        st.write(float(risk_score))

        st.write("### Risk Category")
        st.success(risk_category)
