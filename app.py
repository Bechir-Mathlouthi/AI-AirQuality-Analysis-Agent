import os
import streamlit as st
from typing import Dict, Optional
from dataclasses import dataclass
import requests
from groq import Groq

# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if 'api_keys' not in st.session_state:
    st.session_state['api_keys'] = {
        'aqicn': os.environ.get("AQICN_API_KEY", ""),
        'groq': os.environ.get("GROQ_API_KEY", "")
    }

# ------------------------------------------------------------------------------
# Data Models
# ------------------------------------------------------------------------------
@dataclass
class UserInput:
    city: str
    state: str
    country: str
    medical_conditions: Optional[str]
    planned_activity: str

# ------------------------------------------------------------------------------
# AQI Analyzer using AQICN API for real-time data extraction
# ------------------------------------------------------------------------------
class AQIAnalyzer:
    def __init__(self, aqicn_key: str) -> None:
        self.aqicn_key = aqicn_key

    def _format_url(self, city: str) -> str:
        # Build the API URL using the city name.
        return f"http://api.waqi.info/feed/{city}/?token={self.aqicn_key}"

    def fetch_aqi_data(self, city: str, state: str, country: str) -> Dict[str, float]:
        # For the AQICN API, we use the city name.
        url = self._format_url(city)
        st.info(f"Accessing URL: {url}")
        try:
            response = requests.get(url, timeout=10)
            data = response.json()
            if data.get('status') != 'ok':
                raise ValueError(f"API returned status: {data.get('status')}")
            # Extract overall AQI
            aqi = data['data'].get('aqi', 0)
            # Extract individual metrics from "iaqi"
            iaqi = data['data'].get('iaqi', {})
            pm25 = iaqi.get('pm25', {}).get('v', 0) if isinstance(iaqi.get('pm25'), dict) else 0
            pm10 = iaqi.get('pm10', {}).get('v', 0) if isinstance(iaqi.get('pm10'), dict) else 0
            # Temperature, humidity, wind speed, and CO might not be provided. Default to 0.
            temperature = iaqi.get('t', {}).get('v', 0) if isinstance(iaqi.get('t'), dict) else 0
            humidity = iaqi.get('h', {}).get('v', 0) if isinstance(iaqi.get('h'), dict) else 0
            wind_speed = iaqi.get('w', {}).get('v', 0) if isinstance(iaqi.get('w'), dict) else 0
            co = iaqi.get('co', {}).get('v', 0) if isinstance(iaqi.get('co'), dict) else 0

            result = {
                'aqi': aqi,
                'pm25': pm25,
                'pm10': pm10,
                'temperature': temperature,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'co': co
            }
            with st.expander("📦 Raw AQICN Data", expanded=True):
                st.json(data)
            return result
        except Exception as e:
            st.error(f"Error fetching AQICN data: {str(e)}")
            return {
                'aqi': 0,
                'pm25': 0,
                'pm10': 0,
                'temperature': 0,
                'humidity': 0,
                'wind_speed': 0,
                'co': 0
            }

# ------------------------------------------------------------------------------
# Health Recommendation Agent using Groq API
# ------------------------------------------------------------------------------
class HealthRecommendationAgent:
    def __init__(self, groq_key: str) -> None:
        self.client = Groq(api_key=groq_key)

    def _create_prompt(self, aqi_data: Dict[str, float], user_input: UserInput) -> str:
        return f"""
Based on the following air quality conditions in {user_input.city}, {user_input.state}, {user_input.country}:
- Overall AQI: {aqi_data['aqi']}
- PM2.5 Level: {aqi_data['pm25']} µg/m³
- PM10 Level: {aqi_data['pm10']} µg/m³
- CO Level: {aqi_data['co']} ppb

Weather conditions:
- Temperature: {aqi_data['temperature']}°C
- Humidity: {aqi_data['humidity']}%
- Wind Speed: {aqi_data['wind_speed']} km/h

User's Context:
- Medical Conditions: {user_input.medical_conditions or 'None'}
- Planned Activity: {user_input.planned_activity}

Provide comprehensive health recommendations covering:
1. The impact of current air quality on health.
2. Necessary safety precautions for the planned activity.
3. Advisability of the planned activity.
4. The best time to conduct the activity.
"""

    def get_recommendations(self, aqi_data: Dict[str, float], user_input: UserInput) -> str:
        prompt = self._create_prompt(aqi_data, user_input)
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile"  # Adjust the model if needed.
        )
        return chat_completion.choices[0].message.content

# ------------------------------------------------------------------------------
# Main Analysis Function
# ------------------------------------------------------------------------------
def analyze_conditions(user_input: UserInput, api_keys: Dict[str, str]) -> str:
    aqi_analyzer = AQIAnalyzer(aqicn_key=api_keys['aqicn'])
    health_agent = HealthRecommendationAgent(groq_key=api_keys['groq'])
    aqi_data = aqi_analyzer.fetch_aqi_data(city=user_input.city, state=user_input.state, country=user_input.country)
    return health_agent.get_recommendations(aqi_data, user_input)

# ------------------------------------------------------------------------------
# Streamlit UI Setup Functions
# ------------------------------------------------------------------------------
def setup_page():
    st.set_page_config(page_title="AQI Analysis Agent", page_icon="🌍", layout="wide")
    st.title("🌍 AQI Analysis Agent")
    st.info("Get personalized health recommendations based on real-time air quality data.")

def render_sidebar():
    with st.sidebar:
        st.header("🔑 API Configuration")
        new_aqicn_key = st.text_input(
            "AQICN API Key",
            type="password",
            value=st.session_state['api_keys']['aqicn'],
            help="Enter your AQICN API key"
        )
        new_groq_key = st.text_input(
            "Groq API Key",
            type="password",
            value=st.session_state['api_keys']['groq'],
            help="Enter your Groq API key"
        )
        if new_aqicn_key and new_aqicn_key != st.session_state['api_keys']['aqicn']:
            st.session_state['api_keys']['aqicn'] = new_aqicn_key
            st.success("✅ AQICN API key updated!")
        if new_groq_key and new_groq_key != st.session_state['api_keys']['groq']:
            st.session_state['api_keys']['groq'] = new_groq_key
            st.success("✅ Groq API key updated!")

def render_main_content():
    st.header("📍 Location Details")
    col1, col2 = st.columns(2)
    with col1:
        city = st.text_input("City", placeholder="e.g., Paris")
        state = st.text_input("State", placeholder="(Optional) e.g., Paris")
    with col2:
        country = st.text_input("Country", value="France", placeholder="e.g., France")

    medical_conditions = st.text_input("Medical Conditions (optional)", placeholder="e.g., asthma, allergies")
    planned_activity = st.text_input("Planned Activity", placeholder="e.g., morning jog for 2 hours")

    if st.button("🔍 Analyze & Get Recommendations"):
        if not (city and planned_activity):
            st.error("Please fill in all required fields.")
        elif not (st.session_state['api_keys']['aqicn'] and st.session_state['api_keys']['groq']):
            st.error("Please provide both your AQICN and Groq API keys in the sidebar.")
        else:
            user_input = UserInput(
                city=city,
                state=state,
                country=country,
                medical_conditions=medical_conditions,
                planned_activity=planned_activity
            )
            with st.spinner("🔄 Analyzing conditions..."):
                recommendations = analyze_conditions(user_input, st.session_state['api_keys'])
                st.success("✅ Analysis completed!")
                st.markdown("### 📦 Recommendations")
                st.markdown(recommendations)
                st.download_button(
                    "💾 Download Recommendations",
                    data=recommendations,
                    file_name=f"aqi_recommendations_{city}_{state}.txt",
                    mime="text/plain"
                )

def main():
    setup_page()
    render_sidebar()
    render_main_content()

if __name__ == "__main__":
    main()
