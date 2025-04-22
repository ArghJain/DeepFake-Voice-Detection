import streamlit as st
import subprocess
import librosa
import numpy as np
import tensorflow as tf
import os
from imageio_ffmpeg import get_ffmpeg_exe

# Load trained deepfake detection model
MODEL_PATH = "deepfake_voice_detector.keras" 
model = tf.keras.models.load_model(MODEL_PATH)

def convert_audio(input_file, output_file):
    try:
        ffmpeg_path = get_ffmpeg_exe()  # Get path to bundled ffmpeg binary
        command = [
            ffmpeg_path,
            "-i", input_file,
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            output_file,
            "-y"
        ]
        print("Running command:", " ".join(command))  # Optional debug
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return output_file
    except subprocess.CalledProcessError as e:
        st.error(f"Audio conversion failed: {e.stderr.decode()}")
        return None

# Function to extract features from audio
def extract_features(file_path):
    audio, _ = librosa.load(file_path, sr=16000)
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
    
    # Pad or truncate to fixed shape
    max_pad_length = 100  # Adjust as needed
    if mfccs.shape[1] < max_pad_length:
        pad_width = max_pad_length - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_length]
    
    return np.expand_dims(mfccs, axis=0)  # Reshape for model input

# Function to predict whether the audio is fake or real
def predict_audio(file_path):
    features = extract_features(file_path)
    prediction = model.predict(features)
    return "Real" if prediction[0][0] > 0.5 else "Fake"

# Streamlit UI
st.title("🔍 Deepfake Voice Detection")
st.write("Upload an audio file and click the button to check whether it's real or deepfake.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    
    # Save uploaded file
    input_path = "temp_input." + uploaded_file.name.split(".")[-1]
    output_path = "converted_audio.wav"
    
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Convert audio format
    converted_file = convert_audio(input_path, output_path)

    # Button to trigger prediction
    if st.button("Detect Fake Audio"):
        if converted_file:
            # Predict
            result = predict_audio(converted_file)
            st.success(f"🎤 Prediction: *{result}*")
        
        # Cleanup
        os.remove(input_path)
        os.remove(output_path)


# Footer
st.markdown("---")
st.markdown("👨‍💻 Developed by ")
