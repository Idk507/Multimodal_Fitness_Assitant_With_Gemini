import streamlit as st
import os
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.llms.base import LLM
from googletrans import Translator
import speech_recognition as sr
from PIL import Image
from typing import Optional, List



load_dotenv()
GOOGLE_API_KEY = "" 
if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY is not set.")
genai.configure(api_key=GOOGLE_API_KEY)
translator = Translator()

# Custom Google LLM compatible with LangChain
class GoogleGenAI(LLM):
    @property
    def _llm_type(self) -> str:
        return "google-generativeai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.candidates[0].content.parts[0].text

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return self._call(prompt, stop)

# Instantiate the custom LLM
google_llm = GoogleGenAI()

# Search Tool
duckduckgo_tool = DuckDuckGoSearchRun()
search_tool = Tool(
    name="DuckDuckGo Search",
    func=duckduckgo_tool.run,
    description="Useful for searching online for product recommendations, reviews, and purchase links."
)

# Initialize LangChain agent
agent = initialize_agent(
    tools=[search_tool],
    llm=google_llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

action_re = re.compile(r'^Action:\s*(\w+)\s*:\s*(.*)$')

# System Prompt
system_prompt = """
You are a fitness assistant and product recommendation agent. You help users with workout plans, dietary advice, motivational quotes, and also recommend live protein supplements and the best diet food hotels/products available online.
Your available actions are:
generate_workout: e.g. generate_workout: Beginner 
suggest_meal: e.g. suggest_meal: Low-carb breakfast 
motivational_quotes: e.g. motivational_quotes:
recommend_protein: e.g. recommend_protein:
recommend_diet_hotels: e.g. recommend_diet_hotels:
"""

# Action Functions
def generate_workout(level: str) -> str:
    return google_llm._call(f"Generate a workout plan for a {level} fitness level.")

def suggest_meal(preferences: str) -> str:
    return google_llm._call(f"Suggest a meal plan with {preferences}.")

def motivational_quotes() -> str:
    return google_llm._call("Give me a motivational quote.")

def recommend_protein() -> str:
    return agent.run("Recommend live protein supplements available online with detailed product information, nutritional facts, prices, and purchase links.")

def recommend_diet_hotels() -> str:
    return agent.run("Recommend the best diet food hotels or healthy food product outlets available online with details and links.")

known_actions = {
    "generate_workout": generate_workout,
    "suggest_meal": suggest_meal,
    "motivational_quotes": motivational_quotes,
    "recommend_protein": recommend_protein,
    "recommend_diet_hotels": recommend_diet_hotels
}

# Chatbot Class
class Chatbot:
    def __init__(self, system: str, memory: ConversationBufferMemory):
        self.system = system
        self.memory = memory
        self.memory.chat_memory.messages.append(SystemMessage(content=system))

    def __call__(self, message: str) -> str:
        self.memory.chat_memory.messages.append(HumanMessage(content=message))
        result = self.execute()
        self.memory.chat_memory.messages.append(AIMessage(content=result))
        return result

    def execute(self) -> str:
        conversation = self.memory.load_memory_variables({})
        prompt = "\n".join([msg.content for msg in conversation["chat_history"]])
        return google_llm._call(prompt)

# Translation Functions
def translate_to_english(text: str, src_lang: str) -> str:
    if src_lang.lower() != "en":
        translated = translator.translate(text, src="auto", dest="en")
        return translated.text
    return text

def translate_from_english(text: str, target_lang: str) -> str:
    if target_lang.lower() != "en":
        translated = translator.translate(text, src="en", dest=target_lang)
        return translated.text
    return text

# Query Processing
def process_query(question: str, target_language: str, memory):
    bot = Chatbot(system_prompt, memory)
    question_in_english = translate_to_english(question, target_language)
    result = bot(question_in_english)
    
    actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
    if actions:
        action, action_input = actions[0].groups()
        action = action.strip()
        action_input = action_input.strip()
        if action in known_actions:
            observation = known_actions[action](action_input) if action_input else known_actions[action]()
            return translate_from_english(observation, target_language)
    return translate_from_english(result, target_language)

# Voice Input
def get_voice_query():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    except sr.RequestError as e:
        return f"Error: {e}"

# Image Query Processing
def process_image_query(image_path: str, user_query: str, target_language: str):
    sample_file = genai.upload_file(path=image_path, display_name="Image")
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    prompt = (
        f"Using the uploaded image and the query '{user_query}', provide fitness-related advice or suggestions. you should reply to the query apart from fitness related query and only should reply to the fitness related query. "
        f"Make sure to provide a detailed response. \n\n"
        "The image shows a fitness-related activity or equipment. Assist the user with workout tips, form correction, or equipment usage guidance."
    )
    response = model.generate_content([sample_file, prompt])
    return translate_from_english(response.text, target_language)

# Streamlit App
st.title("BRATZLIFE - Fitness & Product Recommendation Assistant")
st.write("Welcome! I can assist with workout plans, meal suggestions, motivational quotes, product recommendations, and image-based fitness advice.")

# Session State for Memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Sidebar: Chat History
with st.sidebar:
    st.subheader("Chat History")
    for chat in st.session_state.chat_history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['assistant']}")
        st.markdown("---")

# Main Area: Input and Image Upload
col1, col2 = st.columns([2, 1])  # Left for text/voice input, right for image upload

# Left Column: Text and Voice Input
with col1:
    st.subheader("Chat Interface")
    # Language Selection
    target_language = st.selectbox("Select Language", ["en", "es", "fr", "de"], index=0)

    # Input Mode
    input_mode = st.radio("Choose Input Mode", ["Text", "Voice"])

    if input_mode == "Text":
        user_input = st.text_input("Enter your query:")
        if st.button("Submit") and user_input:
            with st.spinner("Processing..."):
                response = process_query(user_input, target_language, st.session_state.memory)
                st.session_state.chat_history.append({"user": user_input, "assistant": response})
                st.success("Done!")
                st.write(f"**Assistant:** {response}")

    elif input_mode == "Voice":
        if st.button("Record Voice Query"):
            with st.spinner("Listening..."):
                voice_query = get_voice_query()
                st.write(f"Recognized: {voice_query}")
                if "Error" not in voice_query and "Sorry" not in voice_query:
                    response = process_query(voice_query, target_language, st.session_state.memory)
                    st.session_state.chat_history.append({"user": voice_query, "assistant": response})
                    st.write(f"**Assistant:** {response}")
                else:
                    st.error(voice_query)
                st.success("Done!")

# Right Column: Image Upload and Query
with col2:
    st.subheader("Fitness Image Assistance")
    uploaded_file = st.file_uploader("Upload a fitness image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        user_query = st.text_input("Enter your query related to this image", key="image_query")
        if st.button("Submit Image Query") and user_query:
            with st.spinner("Processing image query..."):
                # Save uploaded file temporarily
                temp_file_path = f"temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                response = process_image_query(temp_file_path, user_query, target_language)
                st.markdown(f"> {response}")
                # Clean up temporary file
                os.remove(temp_file_path)
                st.session_state.chat_history.append({"user": f"Image Query: {user_query}", "assistant": response})
                st.success("Done!")
