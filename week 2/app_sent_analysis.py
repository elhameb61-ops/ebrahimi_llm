import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import plotly.graph_objects as go
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ================= CONFIG =================
OPENROUTER_API_KEY = "YOUR_OPENROUTER_KEY"

LLM = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="" ,
    model="openai/gpt-oss-120b:free"
)

THRESHOLD = 0.1

class PersianVocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>":0, "<UNK>":1}

    def tokenize(self, text):
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        return text.split()

    def numericalize(self, text, max_len=50):
        words = self.tokenize(text)
        ids = [self.word2idx.get(w, 1) for w in words]
        ids = ids[:max_len]
        ids += [0] * (max_len - len(ids))
        return ids
    
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, emb=64, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden*2, 3)

    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)
    
class Predictor:
    def __init__(self, model, vocab):
        self.model = model
        self.vocab = vocab

    def predict(self, text):
        x = torch.tensor([self.vocab.numericalize(text)])
        out = self.model(x)
        prob = torch.softmax(out, dim=1)[0].detach().numpy()

        return {
            "good": float(prob[0]),
            "neutral": float(prob[1]),
            "bad": float(prob[2]),
            "is_bad": prob[2] > THRESHOLD
        }
    
def check_llm(text):
    messages = [
        SystemMessage(content="""
        You are a sentiment classifier.
        Return JSON:
        {"is_bad": true/false, "confidence": 0-1}
        """),
        HumanMessage(content=text)
    ]

    res = LLM.invoke(messages)

    try:
        return eval(res.content)
    except:
        return {"is_bad": False, "confidence": 0.0}
    
def run_pipeline(text, predictor):
    ml = predictor.predict(text)

    result = {
        "ml": ml,
        "llm": None,
        "final": "OK"
    }

    if ml["is_bad"]:
        llm = check_llm(text)
        result["llm"] = llm

        if llm.get("is_bad") and llm.get("confidence", 0) > 0.6:
            result["final"] = "BAD"
        else:
            result["final"] = "OK"

    return result

st.title("Sentiment Analysis (OpenRouter + LangChain)")

text = st.text_area("Enter Persian text")

if st.button("Analyze"):
    vocab = PersianVocabulary()
    model = SentimentModel(vocab_size=1000)
    predictor = Predictor(model, vocab)

    result = run_pipeline(text, predictor)

    st.subheader("ML Result")
    st.json(result["ml"])

    if result["llm"]:
        st.subheader("LLM Check")
        st.json(result["llm"])

    st.subheader("Final Decision")
    st.success(result["final"])

