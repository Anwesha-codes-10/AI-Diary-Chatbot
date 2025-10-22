from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import random
import json

emotional_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
emotional_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
chat_tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
chat_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
emotional_model.to(device)
chat_model.to(device)

with open("custom_emotion_dictionary.json", "r") as file:
    emotion_labels = json.load(file)

id2label = emotional_model.config.id2label 

def map_emotion(predicted_label):
    for category, label_list in emotion_labels.items():
        if predicted_label.lower() in [x.lower() for x in label_list]:
            return category 
    return predicted_label  


def getmood(text):
    inputs = emotional_tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = emotional_model(**inputs)
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1)
    confidence, predicted_class = torch.max(probabilities, dim=1)
    
    raw_label = id2label[predicted_class.item()]  


    mood = map_emotion(raw_label)
    confidence_score = confidence.item()

    return mood, confidence_score

def generate_response(user_input, chat_history, max_turns=5):
    recent_history = chat_history[-max_turns:]
    history = ""
    for turn in chat_history:
        history += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    prompt = f"{history}User: {user_input}\nBot:"
    
    inputs = chat_tokenizer(prompt,return_tensors="pt").to(device)
    outputs = chat_model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True,

    )

    response = chat_tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in response:
        response = response.replace(prompt," ").strip()
    return response

