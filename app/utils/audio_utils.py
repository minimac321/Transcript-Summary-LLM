# audio_functions.py

import streamlit as st
import os
from db.db_functions import insert_audio

def upload_audio_tab():
    st.header("Upload Audio")
    audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    if audio_file is not None:
        file_path = save_uploaded_file(audio_file)
        transcript_id = st.text_input("Enter Transcript ID:")
        if st.button("Upload"):
            insert_audio(file_path, transcript_id)
            st.success("Audio file uploaded successfully")

def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path