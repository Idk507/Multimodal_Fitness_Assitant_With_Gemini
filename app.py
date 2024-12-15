import streamlit as st
import google.generativeai as genai
from PIL import Image
import re
from dotenv import load_dotenv
import os
import speech_recognition as sr
import tempfile
from pydub import AudioSegment
from pydub.playback import play

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
    bot = Chatbot(system_prompt)
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
system_prompt = """ 
You are a fitness assistant. You help users with workout plans, dietary advice, and motivational quotes.
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

# Initialize session state to store chat history and image state
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []
if 'uploaded_image' not in st.session_state:
    st.session_state['uploaded_image'] = None

# Function to add new messages to chat history
def add_to_chat_history(user_message, bot_response):
    st.session_state['chat_history'].append({
        'user': user_message,
        'bot': bot_response
    })

# Display chat history in the sidebar
st.sidebar.title("Chat History")
if st.session_state['chat_history']:
    for idx, chat in enumerate(st.session_state['chat_history']):
        st.sidebar.write(f"User: {chat['user']}")
        st.sidebar.write(f"Bot: {chat['bot']}")
        st.sidebar.write("---")

# Function to convert voice to text
def transcribe_voice(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError as e:
            return f"Could not request results from Google Speech Recognition service; {e}"

# Custom HTML5-based audio recorder using JavaScript
def audio_recorder():
    audio_html = """
    <script>
    const startRecording = () => {
        const chunks = [];
        const rec = new MediaRecorder(window.stream);

        rec.ondataavailable = e => chunks.push(e.data);
        rec.onstop = e => {
            const completeBlob = new Blob(chunks, { type: 'audio/wav' });
            const audioURL = window.URL.createObjectURL(completeBlob);
            document.getElementById('audio').src = audioURL;
        };

        rec.start();
        setTimeout(() => rec.stop(), 10000);  // stop after 3 seconds
    };

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => { window.stream = stream });

    </script>
    <button onclick="startRecording()">Start Recording</button>
    <audio id="audio" controls></audio>
    """
    st.components.v1.html(audio_html)

# Streamlit App interface
st.title("BRATZLIFE - Fitness Assistant")

# Handle voice input
st.subheader("Record Your Voice Query:")
audio_recorder()

# Upload image handling
uploaded_file = st.file_uploader("Upload a fitness image", type=["png", "jpg", "jpeg"])

# Clear current image when a new query is raised
if uploaded_file:
    st.session_state['uploaded_image'] = uploaded_file

# Display image if one is uploaded
if st.session_state['uploaded_image']:
    image = Image.open(st.session_state['uploaded_image'])
    st.image(image, caption='Uploaded Image', use_column_width=True)

# Transcribe voice query if any voice recording is provided
if st.session_state.get('voice_recording'):
    voice_query = transcribe_voice(st.session_state['voice_recording'])
    user_question = voice_query

# Otherwise, accept text input
else:
    user_question = st.text_input("Ask a fitness question (about workouts, meals, or motivation)")

# If a text or voice query is entered
if user_question or uploaded_file:
    st.write("Processing your request...")
    
    # Clear the current image once a query is made
    if uploaded_file:
        st.session_state['uploaded_image'] = None

    try:
        response = query(user_question)
        st.write(response)

        # Add conversation to chat history
        add_to_chat_history(user_question, response)
    except Exception as e:
        st.error(f"Error: {e}")
