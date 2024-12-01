import streamlit as st
import numpy as np
import geocoder
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import requests

# Constants for sound thresholds (in decibels)
HUMAN_THRESHOLD = 85
QUIET_HOURS = (22, 6)  # Quiet hours from 10 PM to 6 AM

# Initialize session state
if "users" not in st.session_state:
    st.session_state["users"] = {}

if "logged_in_user" not in st.session_state:
    st.session_state["logged_in_user"] = None

if "noise_reports" not in st.session_state:
    st.session_state["noise_reports"] = []

if "sound_level" not in st.session_state:
    st.session_state["sound_level"] = 0

# Helper Functions
def register_user(username, password):
    if username in st.session_state["users"]:
        return False  # Username already exists
    st.session_state["users"][username] = password
    return True

def authenticate_user(username, password):
    return st.session_state["users"].get(username) == password

def get_location():
    try:
        g = geocoder.ip('me')
        return g.address, g.latlng
    except Exception as e:
        st.error("Unable to fetch location.")
        return "Unknown", (0, 0)

def detect_sound(audio_data):
    """Simulate sound detection."""
    # Calculate RMS and dB
    rms = np.sqrt(np.mean(audio_data ** 2))
    db = 20 * np.log10(rms + 1e-6)  # Convert to decibels
    return max(db, 0)  # Ensure no negative values

def is_quiet_hour():
    current_hour = datetime.now().hour
    return QUIET_HOURS[0] <= current_hour or current_hour < QUIET_HOURS[1]

def add_report(timestamp, location, sound_level, description):
    st.session_state["noise_reports"].append({
        "Timestamp": timestamp,
        "Location": location,
        "Sound Level (dB)": sound_level,
        "Description": description,
    })

# Streamlit App
st.set_page_config(page_title="Sound Monitor & Safety")
st.title("Sound Monitor & Safety Dashboard")

# Login Dialog
@st.dialog("Login")
def login_dialog():
    # use tabs
    tabs = st.tabs(["Login", "Register"])

    with tabs[0]:
        username = st.text_input("Username", key="username1")
        password = st.text_input("Password", type="password", key="password1")

        if st.button("Login"):
            if authenticate_user(username, password):
                st.session_state["logged_in_user"] = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials!")
    
    with tabs[1]:
        username = st.text_input("Username", key="username2")
        password = st.text_input("Password", type="password", key="password2")

        if st.button("Register"):
            if register_user(username, password):
                st.session_state["logged_in_user"] = username
                st.success("User registered successfully!")
                st.rerun()
            else:
                st.error("Username already exists!")

if not st.session_state["logged_in_user"]:
    login_dialog()

if st.session_state["logged_in_user"]:
    # Tabs for navigation
    tabs = st.tabs(["Home", "Noise Monitoring", "Report Disturbance", "Reports"])

    # Home
    with tabs[0]:
        st.write("Welcome to the Sound Monitor & Safety Dashboard")
        st.image(
            "https://medico-labs.com/wp-content/uploads/2017/03/Noise-pollution.png",
            use_container_width=True,
        )

    # Noise Monitoring
    with tabs[1]:
        st.header("🔊 Noise Monitoring & Alerts")
        audio_input = st.audio_input("Record Sound", help="Click the microphone icon to record and analyze sound")
        if audio_input:
            # Process the audio input
            audio_data = np.frombuffer(audio_input.read(), dtype=np.int16)
            db = detect_sound(audio_data)
            st.info(f"Detected Sound Level: {db:.2f} dB")
            st.session_state["sound_level"] = round(db, 2)

            if db > HUMAN_THRESHOLD:
                st.warning("Dangerous noise levels detected!")
                
                if is_quiet_hour():
                    st.error("Noise detected during restricted hours!")

            else:
                st.success("Sound levels are within safe limits.")

    # Report Disturbance
    with tabs[2]:
        st.header("📲 Report a Noise Disturbance")
        # display recorded sound level, give user option to record sound again
        st.write(f"Detected Sound Level: {st.session_state["sound_level"]:.2f} dB")
        with st.expander("Record Sound Again ?"):
            audio_input = st.audio_input("Record Sound", help="Click the microphone icon to record and analyze sound", key="audio_input_again")
            if audio_input:
                audio_data = np.frombuffer(audio_input.read(), dtype=np.int16)
                db = detect_sound(audio_data)
                st.info(f"Detected Sound Level: {db:.2f} dB")
                st.session_state["sound_level"] = round(db, 2)

        location_type = st.radio("Type of Area (Automatically Selected)", ["Industrial", "Commercial", "Residential", "Silent Zone"], help="Area Type is automatically detected based on location, but you may change it", horizontal=True, index=2)
        # do you want to upload a photo, to support your report and make it more credible?
        with st.expander("Do you want to provide a photo to support your report and make it more credible?"):

            # photo_option = st.segmented_control("Choose an option to provide a photo:", ("Take a photo", "Upload a photo"), default="Take a photo")
            photo_option = st.radio("Choose an option to provide a photo", ( "Upload a photo", "Take a photo"), horizontal=True)

            if photo_option == "Upload a photo":
                photo = st.file_uploader("Upload a photo of the noise disturbance", type=["jpg", "jpeg", "png"], help="Upload a photo of the noise disturbance")
            else:
                photo = st.camera_input("Take a photo of the noise disturbance", help="Click a photo of the noise disturbance")
        description = st.text_area("Describe the Noise Disturbance", placeholder="Enter a brief description of the noise disturbance")
        location, coords = get_location()
        st.write(f"Detected Location: {location}")
        if st.button("Submit Report"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db = st.session_state["sound_level"]
            add_report(timestamp, location, db, description)
            st.success("Your report has been submitted successfully!")

    # Reports
    with tabs[3]:
        st.header("Submitted Reports")
        reports_df = pd.DataFrame(st.session_state["noise_reports"])
        if not reports_df.empty:
            for i, row in reports_df.iloc[::-1].iterrows():
                st.write(f"Report {len(reports_df) - i}:", row)
        else:
            st.info("No reports submitted yet.")
