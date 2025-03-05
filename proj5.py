import os
import re
import sys
import time
from dotenv import load_dotenv
import google.generativeai as genai
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from diffusers import StableDiffusionPipeline
from PIL import Image
import speech_recognition as sr

# LangChain imports
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, llms
from langchain.llms.base import LLM
from langchain.schema import Generation, LLMResult, SystemMessage, HumanMessage, AIMessage
from langchain.agents import initialize_agent, Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.llms.base import BaseLLM

# Translator for multilingual support
from googletrans import Translator
from typing import Optional, List

os.environ["GOOGLE_API_KEY"] = ""

load_dotenv()
GOOGLE_API_KEY = "AIzaSyCgaz3OFtXuNx-SCRPz2N58UCfpo0pcH_g"
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY is not set.")
genai.configure(api_key=GOOGLE_API_KEY)
translator = Translator()

class GoogleGenAI(LLM):
    @property
    def _llm_type(self) -> str:
        return "google-generativeai"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Create the generative model instance (adjust model name as needed)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        # Extract text from the API response. Adjust extraction if needed.
        return response.candidates[0].content.parts[0].text

    async def _acall(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Delegate async call to the synchronous _call method.
        return self._call(prompt, stop)

# Instantiate our custom LLM
google_llm = GoogleGenAI()

# Instantiate the DuckDuckGo search tool from langchain_community
duckduckgo_tool = DuckDuckGoSearchRun()
search_tool = Tool(
    name="DuckDuckGo Search",
    func=duckduckgo_tool.run,
    description="Useful for searching online for product recommendations, reviews, and purchase links."
)

agent = initialize_agent(
    tools=[search_tool],
    llm=google_llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True
)

action_re = re.compile(r'^Action:\s*(\w+)\s*:\s*(.*)$')

# Updated Chatbot using LangChain message classes directly.
class Chatbot:
    def __init__(self, system: str, memory: ConversationBufferMemory):
        self.system = system
        self.memory = memory
        # Append the system message to memory.
        self.memory.chat_memory.messages.append(SystemMessage(content=system))

    def __call__(self, message: str) -> str:
        # Append the user's message.
        self.memory.chat_memory.messages.append(HumanMessage(content=message))
        result = self.execute()
        # Append the assistant's response.
        self.memory.chat_memory.messages.append(AIMessage(content=result))
        return result

    def execute(self) -> str:
        # Load conversation history as prompt.
        conversation = self.memory.load_memory_variables({})
        prompt = conversation["chat_history"]
        # If prompt is a list (due to return_messages=True), join messages into a single string.
        if isinstance(prompt, list):
            prompt = "\n".join([msg.content for msg in prompt])
        # Generate a response using our custom LLM.
        response = google_llm(prompt)
        return response

system_prompt = """
You are a fitness assistant and product recommendation agent. You help users with workout plans, dietary advice, motivational quotes, and also recommend live protein supplements and the best diet food hotels/products available online.
Your available actions are:
generate_workout:
e.g. generate_workout: Beginner 
Generates a workout plan for the given fitness level.
suggest_meal:
e.g. suggest_meal: Low-carb breakfast 
Suggests a meal plan based on dietary preferences.
motivational_quotes:
e.g. motivational_quotes:
Provides a motivational quote.
recommend_protein:
e.g. recommend_protein:
Recommends live protein supplements with product details and online purchase links.
recommend_diet_hotels:
e.g. recommend_diet_hotels:
Recommends the best diet food hotels or healthy food outlets with details and booking/purchase links.

Example session:
Question: Can you help me with a beginner workout plan?
Thought: I should generate a workout plan.
Action: generate_workout: Beginner

PAUSE 

Observation: Here is a beginner workout plan: 10 pushups, 20 squats, 30 seconds plank.
Answer: I suggest starting with 10 pushups, 20 squats, and 30 seconds plank for your home workout.
""".strip()

def generate_workout(level: str) -> str:
    prompt = f"Generate a workout plan for a {level} fitness level."
    return google_llm(prompt)

def suggest_meal(preferences: str) -> str:
    prompt = f"Suggest a meal plan with {preferences}."
    return google_llm(prompt)

def motivational_quotes() -> str:
    prompt = "Give me a motivational quote."
    return google_llm(prompt)

def recommend_protein() -> str:
    query_text = (
        "Recommend live protein supplements available online with detailed product information, nutritional facts, "
        "prices, and purchase links. Provide a variety of brands."
    )
    result = agent.run(query_text)
    return result

def recommend_diet_hotels() -> str:
    query_text = (
        "Recommend the best diet food hotels or healthy food product outlets available online. Provide details such as location, "
        "menu, pricing, reviews, and booking/purchase links."
    )
    result = agent.run(query_text)
    return result

known_actions = {
    "generate_workout": generate_workout,
    "suggest_meal": suggest_meal,
    "motivational_quotes": motivational_quotes,
    "recommend_protein": recommend_protein,
    "recommend_diet_hotels": recommend_diet_hotels
}

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

def query(question: str, target_language: str = "en", max_turns: int = 5):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    bot = Chatbot(system_prompt, memory)
    
    # Translate query to English if needed.
    question_in_english = translate_to_english(question, target_language)
    next_prompt = question_in_english

    for i in range(max_turns):
        print(f"\nTurn {i+1} - User Query (in English): {next_prompt}")
        result = bot(next_prompt)
        print("\nAssistant raw response (in English):")
        print(result)
        # Look for an action command in the response.
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()
            action = action.strip()
            action_input = action_input.strip()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            print(f"\n-- Executing action: {action} with input: {action_input}")
            observation = known_actions[action](action_input) if action_input else known_actions[action]()
            print(f"\nObservation (in English): {observation}\n")
            next_prompt = f"Answer: {observation}"
            time.sleep(1)
        else:
            translated_response = translate_from_english(result, target_language)
            print("\nAssistant response (translated):")
            print(translated_response)
            return

def get_voice_query() -> Optional[str]:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    print("Please speak your query now...")
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        voice_text = recognizer.recognize_google(audio)
        print(f"Recognized voice query: {voice_text}")
        return voice_text
    except sr.UnknownValueError:
        print("Sorry, could not understand the audio.")
    except sr.RequestError as e:
        print(f"Error from the speech recognition service: {e}")
    return None

def handle_image_query(target_language: str = "en"):
    image_path = input("Enter the path to the image file: ").strip()
    if not os.path.isfile(image_path):
        print("File not found. Please check the path and try again.")
        return
    try:
        image = Image.open(image_path)
        image.show()  # Opens the image using the default viewer
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    user_query = input("Enter your query related to this image: ").strip()
    user_query_in_english = translate_to_english(user_query, target_language)
    try:
        sample_file = genai.upload_file(path=image_path, display_name="Image")
    except Exception as e:
        print(f"Error uploading image: {e}")
        return

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    prompt = (
        f"Using the uploaded image and the query '{user_query_in_english}', provide fitness-related advice or product recommendations. "
        "The image shows a fitness-related activity or equipment. Assist the user with workout tips, form correction, or product recommendations including links."
    )
    response = model.generate_content([sample_file, prompt])
    response_text = response.text
    translated_response = translate_from_english(response_text, target_language)
    print("\nAssistant response for image query (translated):")
    print(translated_response)

print("Welcome to the Multimodal Fitness & Product Recommendation Assistant!")
print("You can query via text, voice, or image.")
target_language = input("Enter your preferred language code (default is 'en'): ").strip().lower() or "en"

while True:
    print("\nChoose query mode:")
    print("1 - Text query")
    print("2 - Voice query")
    print("3 - Image query")
    print("q - Quit")
    choice = input("Enter your choice: ").strip().lower()

    if choice == "1":
        user_query = input("Enter your text query: ").strip()
        query(user_query, target_language=target_language)
    elif choice == "2":
        voice_query = get_voice_query()
        if voice_query:
            query(voice_query, target_language=target_language)
    elif choice == "3":
        handle_image_query(target_language=target_language)
    elif choice == "q":
        print("Exiting application.")
        sys.exit(0)
    else:
        print("Invalid choice. Please select 1, 2, 3, or q.")
