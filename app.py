import os
from datetime import datetime

import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(page_title="MyMind - AI Diary Companion", layout="centered")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

DATA_FILE = "user_data.csv"

MOOD_COLORS = {
    "happy": "#cfe8b8", "sadness": "#a9c3cf", "anger": "#c28b8b", "fear": "#b0aac2",
    "love": "#e6c9d0", "surprise": "#d9d4a6", "disgust": "#a2b39d", "neutral": "#dcdcdc",
    "productive": "#b7d3c7", "confused": "#c7c1d6", "embarrassment": "#deb5b5",
    "hope": "#b8d8c3", "relief": "#c3d2e2", "curiosity": "#d3c3e3", "boredom": "#cfcfcf",
    "guilt": "#bcaac2", "envy": "#b4cfa1", "pride": "#d9c28f", "trust": "#b7d0e8",
    "anxiety": "#bfcbd2", "nostalgia": "#edd8b4", "excitement": "#f1bfae",
    "contentment": "#c9e4c5", "frustration": "#d1a3a4", "disappointment": "#aab2bd",
    "serenity": "#c4dfe6", "enthusiasm": "#f0d0b9", "admiration": "#f2e0b9",
    "loneliness": "#a9a9a9", "vulnerability": "#f0c6cc", "satisfaction": "#c1e1c1",
    "anticipation": "#cdb4db", "determination": "#95b8d1", "joy": "#fde68a"
}

def getmood(text: str):
    t = text.lower()
    rules = [
        ("happy", ["happy", "great", "good", "joy", "excited", "grateful"]),
        ("sadness", ["sad", "down", "unhappy", "depressed", "blue"]),
        ("anger", ["angry", "mad", "furious", "annoyed", "irritated"]),
        ("anxiety", ["anxious", "nervous", "worried", "stressed"]),
        ("love", ["love", "loved", "cared", "affection"]),
        ("surprise", ["surprised", "shocked", "amazed"]),
        ("fear", ["scared", "afraid", "frightened"]),
        ("contentment", ["content", "calm", "peaceful"]),
        ("frustration", ["frustrated", "stuck"]),
    ]
    for label, words in rules:
        if any(w in t for w in words):
            return label, 0.7
    return "neutral", 0.5

FINETUNED_DIR = "./fine_tuned_phi2_qlora"
BASE_ID = "microsoft/phi-2"

@st.cache_resource(show_spinner=True)
def load_model_and_tokenizer():
    model_path = FINETUNED_DIR if os.path.isdir(FINETUNED_DIR) else BASE_ID
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    mdl = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,      
        trust_remote_code=True,
        device_map={"": "cpu"}          
    )
    mdl.eval()
    return tok, mdl, (model_path == FINETUNED_DIR)

tokenizer, model, using_finetuned = load_model_and_tokenizer()


with st.sidebar:
    st.markdown("**Model:** " + ("Fine-tuned Phi-2 ✅" if using_finetuned else "Base Phi-2"))
    st.caption(f"Path: {'./fine_tuned_phi2_qlora' if using_finetuned else 'microsoft/phi-2'}")
    temp = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    max_new = st.slider("Max new tokens", 32, 512, 180, 8)
    rep_pen = st.slider("Repetition penalty", 1.0, 2.0, 1.1, 0.01)

def build_prompt(user_text: str, history: list[dict], max_turns: int = 4) -> str:
    system = "You are MyMind, a warm, concise, empathetic AI diary companion."
    lines = [f"<s>[SYSTEM] {system} [/SYSTEM]"]
    for turn in history[-max_turns:]:
        lines.append(f"[USER] {turn['user']} [/USER]")
        lines.append(f"[ASSISTANT] {turn['bot']} [/ASSISTANT]")
    lines.append(f"[USER] {user_text} [/USER]\n[ASSISTANT]")
    return "\n".join(lines)

@torch.inference_mode()
def generate_response(user_text: str, history: list[dict]) -> str:
    prompt = build_prompt(user_text, history)
    inputs = tokenizer(prompt, return_tensors="pt")
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temp,
        top_p=top_p,
        repetition_penalty=rep_pen,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    full = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return full.split("[ASSISTANT]")[-1].strip() if "[ASSISTANT]" in full else full.strip()

st.title("🧠 MyMind - AI Diary Companion")
user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Get Response"):
    if not user_input.strip():
        st.warning("Please write something first.")
    else:
        mood, conf = getmood(user_input)
        reply = generate_response(user_input, st.session_state.chat_history)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


        st.session_state.chat_history.append(
            {"user": user_input, "bot": reply, "mood": mood, "confidence": round(conf, 2), "timestamp": ts}
        )

        
        row = {"timestamp": ts, "user_input": user_input, "mood": mood, "confidence": round(conf, 2), "response": reply}
        if os.path.exists(DATA_FILE):
            old = pd.read_csv(DATA_FILE)
            pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(DATA_FILE, index=False)
        else:
            pd.DataFrame([row]).to_csv(DATA_FILE, index=False)


if st.session_state.chat_history:
    last = st.session_state.chat_history[-1]["mood"]
    bg = MOOD_COLORS.get(last, "#ffffff")
    st.markdown(f"<style>.stApp {{ background-color: {bg}; }}</style>", unsafe_allow_html=True)

st.subheader("Conversation")
for turn in st.session_state.chat_history:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Bot:** {turn['bot']}")
    st.markdown(
        f"<small>Mood: {turn['mood']} | Confidence: {turn['confidence']} | Timestamp: {turn['timestamp']}</small>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
