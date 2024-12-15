import streamlit as st
import google.generativeai as genai
import re
from dotenv import load_dotenv
import os
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

# Action regex to identify actions in the chatbot response
action_re = re.compile(r'^Action:\s*(\w+)\s*:\s*(.+)$')

# Define the chatbot class and action functions
class Chatbot:
    def __init__(self, system):
        self.system = system
        self.messages = []
        if self.system:
            self.messages.append({"role": "system", "content": system})

    def __call__(self, message):
        self.messages.append({"role": "user", "content": message})
        result = self.execute()
        self.messages.append({"role": "assistant", "content": result})
        return result

    def execute(self):
        prompt = "\n".join([f'{msg["role"]}:{msg["content"]}' for msg in self.messages])
        model = genai.GenerativeModel("gemini-1.5-flash")
        raw_response = model.generate_content(prompt)
        result_text = raw_response.candidates[0].content.parts[0].text
        return result_text

# Action functions using Gemini API
def generate_workout(level):
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"Generate a workout plan for a {level} fitness level")
    return response.candidates[0].content.parts[0].text

def suggest_meal(preferences):
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content(f"Suggest a meal plan with {preferences}")
    return response.candidates[0].content.parts[0].text

def motivational_quotes():
    response = genai.GenerativeModel("gemini-1.5-flash").generate_content("Give me a motivational quote.")
    return response.candidates[0].content.parts[0].text

# Known actions mapping
known_actions = {
    "generate_workout": generate_workout,
    "suggest_meal": suggest_meal,
    "motivational_quotes": motivational_quotes
}

# Query function that interacts with the chatbot
def query(question, max_turns=5):
    i = 0
    bot = Chatbot(prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            observation = known_actions[action](action_input.strip())
            next_prompt = f"Answer: {observation}"
        else:
            return result

# Define the system prompt
prompt = """ 
You are a fitness assistant. You help users with workout plans, dietary advice, and motivational quotes.
You should not answer any other questions other than fitness and diet. Your knowledge and insights are limited to fitness, diet plan, and a healthy lifestyle.
If the user asks any question unrelated to fitness, you should respond with: "I am a Fitness Assistant. I would only answer questions related to Fitness, Diet, and HealthCare."
Your available actions are:
generate_workout:
e.g. generate_workout: Beginner 
Generates a workout plan based on the user's fitness level.
suggest_meal:
e.g. suggest_meal: Low-carb breakfast 
Suggests a meal plan based on the user's dietary preferences.
motivational_quotes:
e.g. motivational_quotes:
Returns a motivational quote to inspire the user.
"""

# Streamlit App interface
st.title("BRATZLIFE - Fitness Assistant Chatbot")
st.write("Ask for workout plans, meal suggestions, or motivational quotes!")

# Speech recognition logic
def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Listening...")
        audio = recognizer.listen(source)
        try:
            query = recognizer.recognize_google(audio)
            return query
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
        except sr.RequestError:
            st.error("Error with the Speech Recognition service.")

# Text-to-speech logic
def text_to_speech(text):
    tts = gTTS(text)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    
    # Using pydub to play the sound
    sound = AudioSegment.from_file(audio_bytes, format="mp3")
    play(sound)
    
    return audio_bytes

# Streamlit components for voice input/output
if st.button("Speak"):
    user_question = speech_to_text()
    if user_question:
        st.write(f"User said: {user_question}")
        try:
            response = query(user_question)
            st.write(response)
            
            # Convert response to speech
            audio_bytes = text_to_speech(response)
            st.audio(audio_bytes, format="audio/mp3")
        except Exception as e:
            st.error(f"Error: {e}")
else:
    user_question = st.text_input("Type your question")

# If the user has entered a question manually
if user_question:
    st.write("Processing your request...")
    try:
        response = query(user_question)
        st.write(response)
        
        # Convert response to speech
        audio_bytes = text_to_speech(response)
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.error(f"Error: {e}")
