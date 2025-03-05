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

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY is not set.")
genai.configure(api_key=GOOGLE_API_KEY)

# Regular expression for parsing actions
action_re = re.compile(r'^Action:\s*(\w+)\s*:\s*(.+)$')

# Chatbot class for text-only multi-turn interaction
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
        # Prepare prompt from the conversation history
        prompt = "\n".join([f'{msg["role"]}:{msg["content"]}' for msg in self.messages])
        model = genai.GenerativeModel("gemini-1.5-flash")
        raw_response = model.generate_content(prompt)
        # Extract text content from the first candidate
        result_text = raw_response.candidates[0].content.parts[0].text
        return result_text

# Sample system prompt defining the fitness assistant and available actions
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

Example session:
Question: Can you help me with a beginner workout plan?
Thought: I should generate a workout plan.
Action: generate_workout: Beginner

PAUSE 

Observation: Here is a beginner workout plan: 10 pushups, 20 squats, 30 seconds plank.
Answer: I suggest starting with 10 pushups, 20 squats, and 30 seconds plank for your home workout.
""".strip()

# Action functions using the Gemini API
def generate_workout(level):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Generate a workout plan for a {level} fitness level")
    return response.candidates[0].content.parts[0].text 

def suggest_meal(preferences):
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(f"Suggest a meal plan with {preferences}")
    return response.candidates[0].content.parts[0].text

def motivational_quotes():
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content("Give me a motivational quote.")
    return response.candidates[0].content.parts[0].text

# Mapping actions to functions
known_actions = {
    "generate_workout": generate_workout,
    "suggest_meal": suggest_meal,
    "motivational_quotes": motivational_quotes
}

# Function for text-based multi-turn conversation
def query(question, max_turns=5):
    i = 0
    bot = Chatbot(system_prompt)
    next_prompt = question
    while i < max_turns:
        i += 1
        result = bot(next_prompt)
        print("\nAssistant response:")
        print(result)
        # Look for an action command in the response
        actions = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            print(f"\n-- running action: {action} with input: {action_input}")
            observation = known_actions[action](action_input.strip())
            print(f"\nObservation: {observation}\n")
            next_prompt = f"Answer: {observation}"
            # Pause briefly before the next turn
            time.sleep(1)
        else:
            # No further action, so end conversation
            return

# Function to capture voice input from the microphone and return recognized text
def get_voice_query():
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
        print(f"Could not request results from the speech recognition service; {e}")
    return None

# Function to handle image queries.
# The user provides a path to an image and an optional accompanying text query.
def handle_image_query():
    image_path = input("Enter the path to the image file: ").strip()
    if not os.path.isfile(image_path):
        print("File not found. Please check the path and try again.")
        return
    try:
        image = Image.open(image_path)
        image.show()  # This will open the image using the default image viewer
    except Exception as e:
        print(f"Error opening image: {e}")
        return

    user_query = input("Enter your query related to this image: ").strip()
    # Upload the file to Google Generative AI (if supported)
    try:
        sample_file = genai.upload_file(path=image_path, display_name="Image")
    except Exception as e:
        print(f"Error uploading image: {e}")
        return

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    # Create a prompt that includes both the query and the reference to the uploaded image
    prompt = (f"Using the uploaded image and the query '{user_query}', provide fitness-related advice or suggestions. "
              "The image shows a fitness-related activity or equipment. Assist the user with workout tips, form correction, "
              "or equipment usage guidance.")
    response = model.generate_content([sample_file, prompt])
    print("\nAssistant response for image query:")
    print(response.text)

# Main function to choose the query modality
def main():
    while True:
        print("\nChoose query mode:")
        print("1 - Text query")
        print("2 - Voice query")
        print("3 - Image query")
        print("q - Quit")
        choice = input("Enter your choice: ").strip().lower()

        if choice == "1":
            user_query = input("Enter your text query: ").strip()
            query(user_query)
        elif choice == "2":
            voice_query = get_voice_query()
            if voice_query:
                query(voice_query)
        elif choice == "3":
            handle_image_query()
        elif choice == "q":
            print("Exiting application.")
            sys.exit(0)
        else:
            print("Invalid choice. Please select 1, 2, 3 or q.")

if __name__ == "__main__":
    main()
