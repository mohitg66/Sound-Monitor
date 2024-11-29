import streamlit as st
import sounddevice as sd
import numpy as np
import geopy
import geocoder
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
# Constants for sound thresholds (in decibels)
HUMAN_THRESHOLD = 85
BABY_THRESHOLD = 70
DOG_THRESHOLD = 80
CAT_THRESHOLD = 70
BIRD_THRESHOLD = 60
QUIET_HOURS = (22, 6)  # Quiet hours from 10 PM to 6 AM

# Initialize local storage for user data
if "users" not in st.session_state:
    st.session_state["users"] = {}

if "logged_in_user" not in st.session_state:
    st.session_state["logged_in_user"] = None

if "noise_reports" not in st.session_state:
    st.session_state["noise_reports"] = []

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

def detect_sound():
    duration = 2  # seconds
    samplerate = 44100
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()
    rms = np.sqrt(np.mean(recording**2))
    db = 20 * np.log10(rms + 1e-6)  # Convert to decibels, prevent log(0) with small epsilon

    # Adjust dB relative to a reference value (e.g., 0 dB = 20 µPa)
    reference_pressure = 20e-6  # Reference pressure in Pascals
    calibrated_db = 20 * np.log10(rms / reference_pressure + 1e-6)

    return max(calibrated_db, 0)  # Ensure no negative values

def find_nearest_police_station(coords):
    """
    Finds the nearest police stations using OpenStreetMap Nominatim API.
    :param coords: Tuple containing latitude and longitude.
    :return: List of nearby police stations.
    """
    try:
        latitude, longitude = coords
        url = f"https://nominatim.openstreetmap.org/search"
        params = {
            "q": "police station",
            "format": "json",
            "addressdetails": 1,
            "limit": 5,
            "lat": latitude,
            "lon": longitude,
        }
        response = requests.get(url, params=params)
        if response.status_code == 200:
            stations = response.json()
            if not stations:
                return "No police stations found nearby."
            station_details = [
                f"{station['display_name']} (Lat: {station['lat']}, Lon: {station['lon']})"
                for station in stations
            ]
            return station_details
        else:
            return "Error fetching data from Nominatim API."
    except Exception as e:
        return f"Error: {str(e)}"
def is_quiet_hour():
    current_hour = datetime.now().hour
    return QUIET_HOURS[0] <= current_hour or current_hour < QUIET_HOURS[1]

def find_nearest_police_station(location):
    # Replace with actual API call to Google Maps or OpenStreetMap
    return "Nearest Police Station: Mock Station, Mock Address"

def notify_police(station, db):
    st.success(f"Notification sent to {station} about loud noise level: {db} dB")

def add_report(description, location, timestamp):
    st.session_state["noise_reports"].append({"Description": description, "Location": location, "Timestamp": timestamp})

# Streamlit App
st.set_page_config(page_title="Sound Monitor & Safety", layout="wide")
st.title("Sound Monitor & Safety Dashboard")

# Authentication
auth_choice = st.sidebar.selectbox("Login/Register", ["Login", "Register"])
if auth_choice == "Register":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Register"):
        if register_user(username, password):
            st.sidebar.success("User registered successfully!")
            st.session_state["logged_in_user"] = username
        else:
            st.sidebar.error("Username already exists!")
elif auth_choice == "Login":
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if authenticate_user(username, password):
            st.sidebar.success("Logged in successfully!")
            st.session_state["logged_in_user"] = username
        else:
            st.sidebar.error("Invalid credentials!")



def detect_frequency():
    """
    Detects sound from the microphone and performs FFT to calculate amplitude vs frequency.
    """
    # Sampling parameters
    sample_rate = 44100  # Sample rate (Hz)
    duration = 3  # seconds to record
    channels = 1  # Mono audio

    # Record audio from the microphone
    st.write("Recording sound...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    st.success("Recording complete!")

    # Perform FFT on the audio data to convert it from time domain to frequency domain
    fft_data = np.abs(np.fft.fft(audio_data.flatten()))  # Amplitude
    fft_freqs = np.fft.fftfreq(len(fft_data), 1 / sample_rate)  # Frequency bins

    # Only consider the positive frequencies for plotting
    positive_freqs = fft_freqs[:len(fft_freqs) // 2]
    positive_fft_data = fft_data[:len(fft_data) // 2]

    return positive_freqs, positive_fft_data


def show_home_page():
    st.header("Welcome to the Sound Monitor & Safety Dashboard")
        
    # Add a Google image from the internet
    image_url = "https://medico-labs.com/wp-content/uploads/2017/03/Noise-pollution.png"  # Replace with the actual image URL
    st.image(image_url, caption="Sound Monitoring & Safety", width=500)  # Set width to desired size


# Main Interface
if st.session_state["logged_in_user"]:
    st.sidebar.header("📍 Location Detection")
    location, coords = get_location()
    st.sidebar.write(f"Current Location: {location}")
    try:
        st.sidebar.map(data={"lat": [coords[0]], "lon": [coords[1]]}, height=220)
    except Exception as e:
        st.sidebar.error("Unable to display map.")


    # Navigation for Features
    feature = st.selectbox("Choose a feature from the dropdown to get started", 
                           ["🏠 Home", "🔊 Noise Monitoring", "📲 Report Disturbance", "📊 Frequency Amplitude Analysis","🎵 Music Loudness Monitoring"])

    if feature == "🏠 Home":
        st.header("Welcome to the Sound Monitor & Safety Dashboard")
        

        # Add a Google image from the internet
        image_url = "https://medico-labs.com/wp-content/uploads/2017/03/Noise-pollution.png"  # Replace with the actual image URL
        st.image(image_url, caption="Sound Monitoring & Safety", width=500)  # Set width to desired size
    elif feature == "🔊 Noise Monitoring":
        st.header("🔊 Noise Monitoring & Alerts")
        if st.button("Start Sound Detection"):
            db = detect_sound()
            st.session_state["current_sound_level"] = db  # Store the sound level in session state
            st.write(f"Detected Sound Level: {db:.2f} dB")
            if db > HUMAN_THRESHOLD:
                st.warning("Dangerous noise levels detected!")
                if is_quiet_hour():
                    st.error("Noise detected during restricted hours! Alerting authorities...")
                    station = find_nearest_police_station(location)
                    notify_police(station, db)
            else:
                st.success("Sound levels are within safe limits.")
            
                        # Bar Graph Display
            st.subheader("Decibel Level Comparison with Current Level")

            # Define safe levels
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
            ax.set_title("Comparison of Safe Levels vs. Your Current Level")
            ax.legend(loc="upper right")
            ax.set_ylim(min(db - 10, 0), max(max(values) + 10, db + 10))  # Adjust y-axis dynamically

            st.pyplot(fig)


    elif feature == "📲 Report Disturbance":
        st.header("📲 Report a Noise Disturbance")
        
        # Step 1: Select the location type
        location_type = st.selectbox("Select the type of area you are currently in:", 
                                    ["Industrial", "Commercial", "Residential", "Silent Zone"])

        # Define noise limits for different zones
        noise_limits = {
            "Industrial": {"Day": 75, "Night": 70},
            "Commercial": {"Day": 65, "Night": 55},
            "Residential": {"Day": 55, "Night": 45},
            "Silent Zone": {"Day": 50, "Night": 40},
        }

        # Display the noise limits for the selected location type
        if location_type:
            limits = noise_limits[location_type]
            st.info(f"{location_type} Area Noise Limits:\n"
                    f"Daytime: {limits['Day']} dB | Nighttime: {limits['Night']} dB")

        # Display the last recorded sound level
        if "current_sound_level" in st.session_state:
            current_db = st.session_state["current_sound_level"]
            st.info(f"**Your Current Sound Level:** {current_db:.2f} dB")
        else:
            st.warning("No sound level detected yet. Please go to the 'Noise Monitoring' page to start sound detection.")
            st.stop()  # Stop execution if no sound level is available

        # Determine the current noise limit based on the time of day
        current_hour = datetime.now().hour
        if 6 <= current_hour < 22:
            noise_limit = limits["Day"]  # Daytime limit
        else:
            noise_limit = limits["Night"]  # Nighttime limit

        # Step 2: Check if the current sound level exceeds the noise limit
        if current_db > noise_limit:
            st.warning(f"⚠️ Your current sound level exceeds the noise limit of {noise_limit} dB!")
            
            # Step 3: Call Police Option
            call_police = st.radio("Do you want to call the police?", ["No", "Yes"], index=0)

            # If "Yes" is selected
            if call_police == "Yes":
                # Map to show nearest police stations
                st.subheader("📍 Nearest Police Stations")
                location, coords = get_location()  # Get address and coordinates
                if coords != (0, 0):  # Check if coordinates are valid
                    st.success(f"Your current location: {location}")
                    st.success(f"Current coordinates: Latitude {coords[0]}, Longitude {coords[1]}")

                    # Fetch and display nearest police stations
                    nearest_stations = find_nearest_police_station(coords)  # Add API logic here
                    if isinstance(nearest_stations, list):
                        st.write("### Nearest Police Stations:")
                        for station in nearest_stations:
                            st.write(f"- {station}")

                    # Display the map
                    try:
                        # Streamlit map requires latitude and longitude columns in a DataFrame
                        map_data = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
                        st.map(map_data, height=300)
                    except Exception as e:
                        st.error(f"Unable to display map. Error: {e}")
                else:
                    st.error("Unable to retrieve your GPS coordinates. Please check your connection.")

                # Option to upload an image as proof
                st.subheader("📷 Attach Proof")
                uploaded_file = st.file_uploader("Upload an image or proof of noise disturbance:", type=["png", "jpg", "jpeg"])
                if uploaded_file is not None:
                    st.image(uploaded_file, caption="Uploaded Proof", use_column_width=True)
                    st.success("Proof uploaded successfully!")

            # Step 4: Describe and submit the report
            description = st.text_area("Describe the Noise Disturbance")
            st.write(f"📍 Detected Location: {location}")
            if st.button("Submit Report"):
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                add_report(description, location, timestamp)
                st.success("Your report has been submitted successfully!")
                st.write("### Submitted Reports")
                reports_df = pd.DataFrame(st.session_state["noise_reports"])
                st.dataframe(reports_df)
        else:
            st.success("Your current sound level is within safe limits. No report required.")
            # Option to calculate the sound level again
            if st.button("🔄 Recalculate Sound Levels"):
                # Simulate recalculating sound levels (Replace this with actual logic to detect sound)
                new_db = detect_sound()  # Replace with your sound level detection function
                st.session_state["current_sound_level"] = new_db
                st.success(f"🔊 New sound level detected: {new_db:.2f} dB")

                        # Step 2: Check if the current sound level exceeds the noise limit
                if new_db > noise_limit:
                    st.warning(f"⚠️ Your current sound level exceeds the noise limit of {noise_limit} dB!")
                    
                    # Step 3: Call Police Option
                    call_police = st.radio("Do you want to call the police?", ["No", "Yes"], index=0)

                    # If "Yes" is selected
                    if call_police == "Yes":
                        # Map to show nearest police stations
                        st.subheader("📍 Nearest Police Stations")
                        location, coords = get_location()  # Get address and coordinates
                        if coords != (0, 0):  # Check if coordinates are valid
                            st.success(f"Your current location: {location}")
                            st.success(f"Current coordinates: Latitude {coords[0]}, Longitude {coords[1]}")

                            # Fetch and display nearest police stations
                            nearest_stations = find_nearest_police_station(coords)  # Add API logic here
                            if isinstance(nearest_stations, list):
                                st.write("### Nearest Police Stations:")
                                for station in nearest_stations:
                                    st.write(f"- {station}")

                            # Display the map
                            try:
                                # Streamlit map requires latitude and longitude columns in a DataFrame
                                map_data = pd.DataFrame({'lat': [coords[0]], 'lon': [coords[1]]})
                                st.map(map_data, height=300)
                            except Exception as e:
                                st.error(f"Unable to display map. Error: {e}")
                        else:
                            st.error("Unable to retrieve your GPS coordinates. Please check your connection.")

                        # Option to upload an image as proof
                        st.subheader("📷 Attach Proof")
                        uploaded_file = st.file_uploader("Upload an image or proof of noise disturbance:", type=["png", "jpg", "jpeg"])
                        if uploaded_file is not None:
                            st.image(uploaded_file, caption="Uploaded Proof", use_column_width=True)
                            st.success("Proof uploaded successfully!")

                    # Step 4: Describe and submit the report
                    description = st.text_area("Describe the Noise Disturbance")
                    st.write(f"📍 Detected Location: {location}")
                    if st.button("Submit Report"):
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        add_report(description, location, timestamp)
                        st.success("Your report has been submitted successfully!")
                        st.write("### Submitted Reports")
                        reports_df = pd.DataFrame(st.session_state["noise_reports"])
                        st.dataframe(reports_df)
    elif feature == "📊 Frequency Amplitude Analysis":
        st.header("🔊 Frequency Analysis of Detected Sounds")

        if st.button("Start Frequency Analysis"):
            # Call the function that processes sound and performs FFT
            freqs, amplitudes = detect_frequency()

            # Display the FFT frequency and amplitude graph
            st.write("### Amplitude vs Frequency Spectrum")
            freq_amplitude_data = pd.DataFrame({
                "Frequency (Hz)": freqs,
                "Amplitude": amplitudes
            })
            st.line_chart(freq_amplitude_data.set_index("Frequency (Hz)"))

            st.success("Frequency analysis complete!")
    elif feature == "🎵 Music Loudness Monitoring":
        st.header("🎵 Real-Time Music Loudness Monitoring")

        # Input for monitoring duration
        duration = st.number_input("Enter monitoring duration (seconds):", min_value=5, max_value=300, value=10, step=5)

        if st.button("Start Monitoring"):
            st.info("Starting monitoring... Please wait.")
            decibel_levels = []  # To store dB levels for the duration
            timestamps = []  # To store time points

            start_time = time.time()  # Get the current time

            while time.time() - start_time < duration:  # Loop for the given duration
                current_db = detect_sound()  # Call the function to measure current dB level
                elapsed_time = int(time.time() - start_time)

                # Record data
                decibel_levels.append(current_db)
                timestamps.append(elapsed_time)

                # Display feedback
                if current_db < 60:
                    st.success(f"[{elapsed_time}s] Safe Level: {current_db:.2f} dB")
                elif 60 <= current_db < 70:
                    st.warning(f"[{elapsed_time}s] Warning Level: {current_db:.2f} dB")
                else:
                    st.error(f"[{elapsed_time}s] Dangerous Level: {current_db:.2f} dB")

                time.sleep(1)  # Pause for 1 second before the next reading

            # Plot the loudness data over time
            st.write("### Loudness Over Time")
            df = pd.DataFrame({"Time (s)": timestamps, "Loudness (dB)": decibel_levels})
            st.line_chart(df.set_index("Time (s)"))

            # Analyze unsafe exposure
            unsafe_duration = sum(1 for db in decibel_levels if db > 85)
            if unsafe_duration > 0:
                st.warning(f"Warning: Unsafe loudness detected for {unsafe_duration} seconds.")
            else:
                st.success("No unsafe loudness detected. Great listening environment!")

                    
            
else:
    st.warning("Please log in to use the app.")
