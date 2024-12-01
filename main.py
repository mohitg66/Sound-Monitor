import streamlit as st
import numpy as np
import geocoder
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import requests
import time
# import plotly.express as px
import soundfile as sf
import io

# Constants for sound thresholds (in decibels)
HUMAN_THRESHOLD = 85
BABY_THRESHOLD = 70
DOG_THRESHOLD = 80
CAT_THRESHOLD = 70
BIRD_THRESHOLD = 60
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
    placeholder = st.empty()
    placeholder.write("Fetching location...")
    time.sleep(2)  # Simulate delay
    placeholder.empty()

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
    
    # Adjust dB relative to a reference value (e.g., 0 dB = 20 µPa)
    reference_pressure = 20e-6  # Reference pressure in Pascals
    calibrated_db = 20 * np.log10(rms / reference_pressure + 1e-6)

    return max(calibrated_db, 0)  # Ensure no negative values

def is_quiet_hour():
    current_hour = datetime.now().hour
    return QUIET_HOURS[0] <= current_hour or current_hour < QUIET_HOURS[1]

def detect_frequency(audio_data):
    # Detects sound from the microphone and performs FFT to calculate amplitude vs frequency.
    audio_data, sample_rate = sf.read(io.BytesIO(audio_data.getvalue()))
    
    # Perform FFT on the audio data to convert it from time domain to frequency domain
    fft_data = np.abs(np.fft.fft(audio_data.flatten()))  # FFT of the audio data
    fft_freqs = np.fft.fftfreq(len(fft_data), 1 / sample_rate)  # Frequency bins

    # Only consider the positive frequencies for plotting
    positive_freqs = fft_freqs[:len(fft_freqs) // 2]
    positive_fft_data = fft_data[:len(fft_data) // 2]

    return positive_freqs, positive_fft_data

def add_report(timestamp, location, sound_level, description):
    st.session_state["noise_reports"].append({
        "Timestamp": timestamp,
        "Location": location,
        "Sound Level (dB)": sound_level,
        "Description": description,
    })

# Streamlit App
st.set_page_config(page_title="Sound Monitor & Safety", page_icon="🔊", layout="wide")
# st.title("Sound Monitor & Safety Dashboard")

# Login Dialog
@st.dialog("Login")
def login_dialog():
    # use tabs
    tabs = st.tabs(["Login", "Register"])

    with tabs[0]:
        with st.form(key='login_form', border=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            login_button = st.form_submit_button("Login")
            if login_button:
                if authenticate_user(username, password):
                    st.session_state["logged_in_user"] = username
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Invalid credentials!")

    with tabs[1]:
        with st.form(key='register_form', border=False):
            username = st.text_input("Username", placeholder="Enter a username")
            password = st.text_input("Password", type="password", placeholder="Enter a password")
            register_button = st.form_submit_button("Register")
            if register_button:
                if register_user(username, password):
                    st.session_state["logged_in_user"] = username
                    st.success("User registered successfully!")
                    st.rerun()
                else:
                    st.error("Username already exists!")

if not st.session_state["logged_in_user"]:
    st.warning("Please login to continue")
    if st.button("Login"):
        st.rerun()
    login_dialog()

if st.session_state["logged_in_user"]:
    # Tabs for navigation
    tabs = st.tabs(["Home", "Noise Monitoring", "Report Disturbance", "Reports", "Frequency Analysis"])

    # Home
    with tabs[0]:
        st.header("🏡 Welcome to the Sound Monitor & Safety Dashboard")
        st.image(
            "image.png",
            use_container_width=False,
        )

    # Noise Monitoring
    with tabs[1]:
        st.header("🔊 Noise Monitoring & Alerts")
        audio_input = st.audio_input("Record Sound", help="Click the microphone icon to record and analyze sound")
        if audio_input:
            # Process the audio input
            audio_data = np.frombuffer(audio_input.read(), dtype=np.int16)
            db = detect_sound(audio_data)
            st.info(f"Detected Sound Level: **{db:.2f} dB**")
            st.session_state["sound_level"] = round(db, 2)

            if db > HUMAN_THRESHOLD:
                st.markdown("<span style='color:red; font-size:20px;'>⚠️ Dangerous noise levels detected!</span>", unsafe_allow_html=True)
                # st.error("Dangerous noise levels detected!")
                if is_quiet_hour():
                    st.markdown("<span style='color:red;'>Noise detected during restricted hours!</span>", unsafe_allow_html=True)
                st.toast("Report the noise disturbance by going to the 'Report Disturbance' tab")
            else:
                st.success("Sound levels are within safe limits.")

            # Bar Graph
            safe_levels = {
                "Humans": HUMAN_THRESHOLD,
                "Dogs": DOG_THRESHOLD,
                "Cats": CAT_THRESHOLD,
                "Birds": BIRD_THRESHOLD,
            }

            labels = list(safe_levels.keys())
            values = list(safe_levels.values())

            fig, ax = plt.subplots(figsize=(8, 5))

            # Plot bar graph for safe levels
            bars = ax.bar(labels, values, color='green', alpha=0.7, label="Safe Levels")

            # Add a vertical line for the current level
            ax.axhline(y=db, color='blue', linestyle='--', linewidth=2, label=f"Your Level: {db:.1f} dB")

            # Add labels for each bar
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height + 2, f"{height} dB", ha='center', fontsize=10)

            # Configure the plot
            ax.set_ylabel("Decibels (dB)")
            ax.set_title("Comparison of Safe Levels vs Your Current Level")
            ax.legend(loc="upper right")
            # ax.set_ylim(min(db - 10, 0), max(max(values) + 10, db + 10))  # Adjust y-axis dynamically

            st.pyplot(fig)
            
        

    # Report Disturbance
    with tabs[2]:
        st.header("📝 Report a Noise Disturbance")
        # display recorded sound level, give user option to record sound again
        st.info(f"Detected Sound Level: **{st.session_state["sound_level"]:.2f} dB**")
        with st.expander("**Record Sound Again ?**"):
            audio_input = st.audio_input("Record Sound", help="Click the microphone icon to record and analyze sound", key="audio_input_again")
            if audio_input:
                audio_data = np.frombuffer(audio_input.read(), dtype=np.int16)
                db = round(detect_sound(audio_data), 2)
                st.info(f"Detected Sound Level: {db:.2f} dB")
                st.session_state["sound_level"] = round(db, 2)

        db = st.session_state["sound_level"]
        location, coords = get_location()
        
        location_type = st.radio("**Type of Area** (Automatically Detected)", ["Industrial", "Commercial", "Residential", "Silent Zone"], help="Area Type is automatically detected based on location, but you may change it", horizontal=True, index=2)

        # display the limits of sound levels for the selected location type
        noise_limits = {
            "Industrial": {"Day": 75, "Night": 70},
            "Commercial": {"Day": 65, "Night": 55},
            "Residential": {"Day": 55, "Night": 45},
            "Silent Zone": {"Day": 50, "Night": 40},
        }

        limits = noise_limits[location_type]
        st.markdown(f'<p style=""><b>{location_type} Area Noise Limits</b><br>'
            f'Daytime: {limits["Day"]} dB | Night: {limits["Night"]} dB</p>', unsafe_allow_html=True)
        
        st.markdown(f'<p style="">Your Current Sound Level: {db:.2f} dB</p>', unsafe_allow_html=True)

        if 6 <= datetime.now().hour < 22:
            noise_limits = limits["Day"]
        else:
            noise_limits = limits["Night"]

        if db > noise_limits:
            st.markdown(f"<p style='color:red;'>Current Sound level <strong>{db:.2f} dB<strong> exceeds the permissible limit of {noise_limits} dB for a {location_type} area</p>", unsafe_allow_html=True)

        # space
        st.write("")

        # do you want to upload a photo, to support your report and make it more credible?
        with st.expander("Include a photo ?"):

            # photo_option = st.segmented_control("Choose an option to provide a photo:", ("Take a photo", "Upload a photo"), default="Take a photo")
            photo_option = st.radio("Choose an option to provide a photo", ( "Upload a photo", "Take a photo"), horizontal=True)

            if photo_option == "Upload a photo":
                photo = st.file_uploader("Upload a photo of the noise disturbance", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            else:
                photo = st.camera_input("Take a photo of the noise disturbance", help="Click a photo of the noise disturbance")
        description = st.text_area("Describe the Noise Disturbance", placeholder="Enter a brief description of the noise disturbance")
        
        st.write(f"Detected Location: Delhi, IN")
        if st.button("Submit Report"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            db = st.session_state["sound_level"]
            add_report(timestamp, location, db, description)
            st.success("Your report has been submitted successfully!")

            reports_df = pd.DataFrame(st.session_state["noise_reports"])
            st.dataframe(reports_df.iloc[-1], use_container_width=True)

    # Reports
    with tabs[3]:
        st.header("📑 Submitted Reports")
        reports_df = pd.DataFrame(st.session_state["noise_reports"])
        if not reports_df.empty:
            for i, row in reports_df.iloc[::-1].iterrows():
                st.write(f"Report {len(reports_df) - i}")
                st.dataframe(row, use_container_width=True)
        else:
            st.info("No reports submitted yet")

    # Frequency Analysis
    with tabs[4]:
        st.header("🔊 Frequency Analysis of Detected Sounds")

        # if st.button("Start Frequency Analysis"):
        # Call the function that processes sound and performs FFT
        audio_data_freq = st.audio_input("Record Sound for Frequency Analysis", help="Click the microphone icon to record sound for frequency analysis")
        if audio_data_freq:
            st.success("Recording complete!")
            freqs, amplitudes = detect_frequency(audio_data_freq)

            # Display the FFT frequency and amplitude graph
            st.write("### Amplitude vs Frequency Spectrum")
            freq_amplitude_data = pd.DataFrame({
                "Frequency (Hz)": freqs,
                "Amplitude": amplitudes
            })
            st.line_chart(freq_amplitude_data.set_index("Frequency (Hz)"), x_label="Frequency (Hz)", y_label="Amplitude", use_container_width=True)

    # Music Loudness Meter
    # with tabs[5]:
    #     st.header("🎵 Real-Time Music Loudness Monitoring")

    #     # Radio button for choosing between upload and record
    #     input_method = st.radio("Choose input method:", ["Upload Audio", "Record Audio"], horizontal=True)

    #     if input_method == "Upload Audio":
    #         audio_data = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    #     else:
    #         audio_data = st.audio_input("Record audio for analysis", key="audio_input_music")

    #     if audio_data is not None:
    #         st.info("Audio received. Analyzing...")
            
    #         # Analyze the audio
    #         if audio_data is not None:
    #             # Read the audio data from the UploadedFile object
    #             audio_samples, sample_rate = sf.read(io.BytesIO(audio_data.getvalue()))
    #             decibel_level = detect_sound(audio_samples)
            
    #         if decibel_level is not None:
    #             # Display the overall decibel level
    #             st.markdown(f"### Overall Decibel Level: <span style='color: red;'>{decibel_level:.2f} dB</span>", unsafe_allow_html=True)
                
    #             # Provide feedback based on the decibel level
    #             if decibel_level < 60:
    #                 st.success(f"Safe Level: {decibel_level:.2f} dB")
    #             elif 60 <= decibel_level < 70:
    #                 st.warning(f"Warning Level: {decibel_level:.2f} dB")
    #             else:
    #                 st.error(f"Dangerous Level: {decibel_level:.2f} dB")
                
    #             # Plot the audio waveform
    #             time = np.linspace(0, len(audio_samples), len(audio_samples))
                
    #             fig = px.line(
    #                 x=time,
    #                 y=audio_samples,
    #                 title="Audio Waveform",
    #                 labels={"x": "Time (seconds)", "y": "Amplitude"},
    #             )
    #             fig.update_layout(
    #                 title_font_size=20,
    #                 xaxis_title_font_size=16,
    #                 yaxis_title_font_size=16,
    #                 font=dict(size=14)
    #             )
    #             st.plotly_chart(fig)
                
    #             # Analyze safety
    #             if decibel_level > 70:
    #                 st.warning("Warning: The audio exceeds safe listening levels.")
    #             else:
    #                 st.success("The audio is within safe listening levels.")
                
    #         else:
    #             st.warning("Unable to analyze the audio. Please try again.")
    #     else:
    #         st.info("Please upload or record audio for analysis.")