import streamlit as st
from textblob import TextBlob
import pandas as pd #We import the pandas library to create and save the data in tabular(CSV) format.
import numpy as np #We import the numpy library to perform numerical operations on the data.
from datetime import datetime #We import the datetime module to work with dates and times.
import os #We import the os module to interact with the computer's files we  will use to check the CSV file if already exists so that we can create a new file or add data to existing one.

st.title("Hey! I'm MyMind - your personal AI diary companion")
st.header("Type anything on your mind. I'm here for you.")
user_input=st.text_input("Type here.")
st.write("Got it. You can share more if you'd like")
analysis = TextBlob(user_input)
if st.button("Submit"):
    if user_input.strip() == "":
        st.write("Empty String!")
    else:
        if analysis.sentiment.polarity > 0.5:
            st.write("🥳You seem really happy today. That's awesome!")
        elif 0.1 < analysis.sentiment.polarity <= 0.5:
            st.write("😊You seem to be in a good and cheerful mood. Keep it up!")
        elif -0.1 <= analysis.sentiment.polarity <= 0.1:
            st.write("🫥A calm Day, huh? That's perfectly okay.")
        elif -0.5 <= analysis.sentiment.polarity < -0.1:
            st.write("😞You seem to be feeling down. It's okay to feel this way.")
        else:
            st.write("🫂Looks like it's been a tough time. Take your time. I'm listening.")


timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
data = {
    "Timestamp": [timestamp],
    "User Input": [user_input],
    "Polarity": [analysis.sentiment.polarity],
    "Subjectivity": [analysis.sentiment.subjectivity]
}

# Check if the CSV file already exists
file_exists = os.path.isfile("user_data.csv")

# If the file exists, append the new data
if file_exists:
    df = pd.read_csv("user_data.csv")
    df = pd.concat([df, pd.DataFrame(data)], ignore_index=True) #pd.DataFrame(data) creates a new DataFrame from the data dictionary, and pd.concat combines the existing DataFrame with the new one.
else:
    # If the file doesn't exist, create a new DataFrame
    df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv("user_data.csv", index=False)

#Showing past diary entries
if st.checkbox("Show past diary entries"):
    if os.path.isfile("user_data.csv"):
        st.subheader("Past Diary Entries")
        past_entries = pd.read_csv("user_data.csv")
        st.dataframe(past_entries)  # Display the DataFrame in Streamlit
    else:
        st.write("No past entries found.")

