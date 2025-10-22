import streamlit as st
from datetime import datetime
import pandas as pd
import os
from chatbot import getmood, generate_response

st.set_page_config(page_title="MyMind - AI Diary Companion", layout="centered")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



data_file = "user_data.csv"

mood_colors = {
    "happy": "#cfe8b8",           # soft green
    "sadness": "#a9c3cf",         # cloudy blue
    "anger": "#c28b8b",           # dusty rose red
    "fear": "#b0aac2",            # muted lavender-gray
    "love": "#e6c9d0",            # muted blush
    "surprise": "#d9d4a6",        # faded yellow-olive
    "disgust": "#a2b39d",         # muted green-gray
    "neutral": "#dcdcdc",         # classic light gray
    "productive": "#b7d3c7",      # gentle mint-teal
    "confused": "#c7c1d6",        # misty violet
    "embarrassment": "#deb5b5",   # pale rose
    "hope": "#b8d8c3",            # soft pastel green
    "relief": "#c3d2e2",          # soft periwinkle
    "curiosity": "#d3c3e3",       # light purple curiosity
    "boredom": "#cfcfcf",         # dull gray
    "guilt": "#bcaac2",           # subdued mauve
    "envy": "#b4cfa1",            # toned-down green
    "pride": "#d9c28f",           # muted gold
    "trust": "#b7d0e8",           # soft baby blue
    "anxiety": "#bfcbd2",         # pale blue-gray
    "nostalgia": "#edd8b4",       # faded beige-warm
    "excitement": "#f1bfae",      # soft coral peach
    "contentment": "#c9e4c5",     # pastel green
    "frustration": "#d1a3a4",     # dusty rose
    "disappointment": "#aab2bd",  # grayish slate
    "serenity": "#c4dfe6",        # pale blue
    "enthusiasm": "#f0d0b9",      # calm orange-beige
    "admiration": "#f2e0b9",      # light cream gold
    "loneliness": "#a9a9a9",      # standard dark gray
    "vulnerability": "#f0c6cc",   # gentle pink
    "satisfaction": "#c1e1c1",    # faded green
    "anticipation": "#cdb4db",    # dusty lilac
    "determination": "#95b8d1",    # steel blue muted
     "joy": "#fde68a"                # soft yellow
}

st.title("🧠 MyMind - AI Diary Companion")
user_input = st.text_area("How are you feeling today?", height=150)

if st.button("Get Response"):
    if user_input.strip() == "":
        st.warning("Please write something first.")
    else:
        mood, confidence_score = getmood(user_input)
        response = generate_response(user_input, st.session_state.chat_history)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        st.session_state.chat_history.append({
            "user": user_input,
            "bot": response,
            "mood": mood,
            "confidence": round(confidence_score, 2),
            "timestamp": timestamp
        })

        log_entry = {
            "timestamp": timestamp,
            "user_input": user_input,
            "mood": mood,
            "confidence": round(confidence_score, 2),
            "response": response
        }

        
        if os.path.exists(data_file):
            old = pd.read_csv(data_file)
            new = pd.DataFrame([log_entry])
            updated = pd.concat([old, new],ignore_index=True)
            updated.to_csv(data_file, index=False)
        else:
            pd.DataFrame([log_entry]).to_csv(data_file, index=False)

    if st.session_state.chat_history:
        last_mood = st.session_state.chat_history[-1]["mood"]
        background_color = mood_colors.get(last_mood, "#ffffff")

        st.markdown(f"""
        <style>
            .stApp {{background-color: {mood_colors[mood]};}}
        </style>
        """, unsafe_allow_html=True)

st.subheader("Conversation")
for turn in st.session_state.chat_history:
    st.markdown(f"**You:** {turn['user']}")
    st.markdown(f"**Bot:** {turn['bot']}")
    st.markdown(f"<small>Mood: {turn['mood']} | Confidence: {turn['confidence']} | Timestamp: {turn['timestamp']}</small>", unsafe_allow_html=True)
    st.markdown("---")