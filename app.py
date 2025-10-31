import asyncio
import inspect
import os
import re
import tempfile
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
try:
    from googletrans import Translator
except ImportError:  # pragma: no cover - optional dependency
    Translator = None  # type: ignore
import speech_recognition as sr
from PIL import Image
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


load_dotenv()

GOOGLE_API_KEY = ""
if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY is not set. Add it to your environment for the assistant to respond.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)


GENERATION_CONFIG = {
    "temperature": 0.3,  # Lower for more focused responses
    "top_p": 0.85,
    "top_k": 20,
    "max_output_tokens": 2048,  # Increased for detailed workout/meal plans
}

MODEL_NAME = "gemini-2.5-flash"


@st.cache_resource(show_spinner=False)
def get_text_model():
    return genai.GenerativeModel(model_name=MODEL_NAME, generation_config=GENERATION_CONFIG)


@st.cache_resource(show_spinner=False)
def get_vision_model():
    return genai.GenerativeModel(model_name="gemini-2.5-flash", generation_config=GENERATION_CONFIG)


translator = Translator() if Translator else None
search_tool = DuckDuckGoSearchRun(api_wrapper=DuckDuckGoSearchAPIWrapper())

ACTION_PATTERN = re.compile(r"^ACTION:\s*(?P<name>[a-z_]+)\s*\|\s*(?P<payload>.*)$", re.IGNORECASE | re.MULTILINE)

FITNESS_SYSTEM_PROMPT = "You are a professional fitness assistant. Provide detailed, comprehensive, and actionable advice."

ACTION_GUIDE = (
    "TOOLS: generate_workout(level), suggest_meal(prefs), motivational_quotes(), recommend_protein(focus), recommend_diet_hotels(location).\n"
    "If user asks for workout/meal/quote/protein/restaurants, respond ONLY: ACTION: tool_name | input\n"
    "Otherwise, provide a detailed, helpful answer with specific information."
)

# All logic handled by LLM - no hardcoded rules


def _describe_finish_reason(response: Any) -> str:
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        reason = getattr(candidate, "finish_reason", None)
        if reason:
            return str(reason)
    return ""


def extract_model_text(response: Any) -> str:
    if response is None:
        return ""
    try:
        text = response.text  # type: ignore[attr-defined]
        if text:
            return str(text).strip()
    except ValueError:
        pass
    except AttributeError:
        pass

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) if content else None
        if parts:
            collected: List[str] = []
            for part in parts:
                text = getattr(part, "text", None)
                if text:
                    collected.append(str(text))
            if collected:
                return "\n".join(collected).strip()

    reason = _describe_finish_reason(response)
    if reason:
        return f"The model stopped early (finish reason: {reason}). Please adjust the prompt and try again."
    return ""


# Removed retry mechanism to optimize response time


def _resolve_translation(result, fallback: str) -> str:
    if result is None:
        return fallback
    if inspect.isawaitable(result) or asyncio.iscoroutine(result):
        try:
            result = asyncio.run(result)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(result)
            finally:
                loop.close()
    if hasattr(result, "text"):
        return getattr(result, "text") or fallback
    if isinstance(result, str):
        return result
    return fallback


def translate_to_english(text: str, source_language: str) -> str:
    if not text or source_language.lower() == "en":
        return text
    return _translate_text(text, src="auto", dest="en")


def translate_from_english(text: str, target_language: str) -> str:
    if not text or target_language.lower() == "en":
        return text
    return _translate_text(text, src="en", dest=target_language)


def _translate_text(text: str, src: str, dest: str) -> str:
    if translator is None:
        return text
    translate_fn = getattr(translator, "translate", None)
    if translate_fn is None:
        return text
    try:
        if inspect.iscoroutinefunction(translate_fn):
            coro = translate_fn(text, src=src, dest=dest)
            return _resolve_translation(coro, text)
        result = translate_fn(text, src=src, dest=dest)
        return _resolve_translation(result, text)
    except Exception:
        return text


def rerun_app() -> None:
    rerun_callable = getattr(st, "rerun", None)
    if callable(rerun_callable):
        rerun_callable()
        return
    experimental = getattr(st, "experimental_rerun", None)
    if callable(experimental):
        experimental()


def generate_text(prompt: str) -> str:
    if not GOOGLE_API_KEY:
        return "I need a valid Google API key before I can reply."
    try:
        model = get_text_model()
        response = model.generate_content(prompt)
        text = extract_model_text(response)
        return text or ""
    except Exception:
        return "I hit a snag while generating that. Try again in a moment."


def compose_prompt(conversation: List[Dict[str, str]], allow_action: bool) -> str:
    history_lines = []
    for message in conversation[-6:]:  # Only last 6 messages to reduce context
        role = message["role"]
        content = message["content"].strip()
        if not content:
            continue
        if role == "user":
            history_lines.append(f"User: {content}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {content}")
        elif role == "tool":
            # Keep full tool output for detailed context
            history_lines.append(f"Data: {content}")

    history_text = "\n".join(history_lines)
    
    if allow_action:
        guidance = f"{FITNESS_SYSTEM_PROMPT}\n{ACTION_GUIDE}"
    else:
        guidance = f"{FITNESS_SYSTEM_PROMPT}\nPresent the complete data above in a clear, formatted way."

    return f"{guidance}\n\n{history_text}\nAssistant:"


def generate_workout(level: str) -> str:
    prompt = (
        f"Create a comprehensive 7-day workout plan for {level or 'general fitness'} level.\n\n"
        f"Format as a detailed markdown table with these columns:\n"
        f"Day | Focus | Warm-up (5-10min) | Main Exercises | Sets x Reps/Duration | Cool-down | Notes\n\n"
        f"Requirements:\n"
        f"- Include 2 full-body strength training days with specific exercises\n"
        f"- Include 2 cardio days with intensity levels and duration\n"
        f"- Include 1 active recovery day (yoga, stretching, light walk)\n"
        f"- Include 2 complete rest days\n"
        f"- For each exercise, specify sets, reps, and rest periods\n"
        f"- Add helpful coaching tips in the Notes column\n"
        f"- Make it practical and equipment-specific (bodyweight, dumbbells, or gym)\n\n"
        f"Provide the complete 7-day breakdown with all details filled in."
    )
    return generate_text(prompt)


def suggest_meal(preferences: str) -> str:
    preferences = preferences or "high-protein"
    prompt = (
        f"Create a detailed one-day {preferences} meal plan.\n\n"
        f"Format as a markdown table with columns:\n"
        f"Meal | Time | Food Items | Portion Size | Calories | Protein(g) | Carbs(g) | Fat(g) | Prep Notes\n\n"
        f"Include:\n"
        f"- Breakfast, Mid-morning snack, Lunch, Afternoon snack, Dinner, Evening snack (if needed)\n"
        f"- Specific ingredients and quantities\n"
        f"- Total daily macros at the bottom\n"
        f"- Quick prep instructions or tips\n\n"
        f"Make it practical and delicious."
    )
    return generate_text(prompt)


def motivational_quotes(_: str = "") -> str:
    prompt = "One short fitness motivation quote. No explanation."
    return generate_text(prompt)


def summarise_search(query: str, task_instruction: str) -> str:
    try:
        snippets = search_tool.run(query)
        # Limit snippets but allow more for detailed responses
        if len(snippets) > 2000:
            snippets = snippets[:2000] + "..."
    except Exception:
        return "Search unavailable. Try again later."
    prompt = f"Based on this data: {snippets}\n\nTask: {task_instruction}\n\nProvide detailed, well-structured information."
    return generate_text(prompt)


def recommend_protein(focus: str = "") -> str:
    query = f"protein supplements {focus} 2024" if focus else "best protein supplements 2024"
    task = (
        "Create a detailed comparison of top 3-5 protein supplements.\n"
        "For each, include: Brand name, Product name, Protein per serving, "
        "Serving size, Key ingredients, Price range, Flavor options, "
        "Best for (goal/use case), Where to buy.\n"
        "Format as a clear comparison with pros and cons."
    )
    return summarise_search(query, task)


def recommend_diet_hotels(location: str = "") -> str:
    query = f"healthy restaurants meal prep {location}" if location else "healthy meal delivery services"
    task = (
        "List 4-5 healthy eating options with details:\n"
        "Name, Type (restaurant/delivery/meal prep), Specialty dishes, "
        "Macro information available, Price range, Ordering method, "
        "Delivery/location details, Best for.\n"
        "Format clearly with all details."
    )
    return summarise_search(query, task)


KNOWN_ACTIONS = {
    "generate_workout": generate_workout,
    "suggest_meal": suggest_meal,
    "motivational_quotes": motivational_quotes,
    "recommend_protein": recommend_protein,
    "recommend_diet_hotels": recommend_diet_hotels,
}


def run_conversation() -> str:
    conversation = st.session_state.conversation
    allow_action = True

    while True:
        prompt = compose_prompt(conversation, allow_action=allow_action)
        raw_response = generate_text(prompt)

        match = ACTION_PATTERN.search(raw_response) if allow_action else None
        if match:
            action_name = match.group("name").lower()
            payload = (match.group("payload") or "").strip()

            if action_name in KNOWN_ACTIONS:
                tool_result = KNOWN_ACTIONS[action_name](payload)
                conversation.append({"role": "assistant", "content": f"ACTION: {action_name} | {payload}"})
                conversation.append({"role": "tool", "content": tool_result})
                allow_action = False
                continue

        response_text = raw_response.strip()
        conversation.append({"role": "assistant", "content": response_text})
        return response_text


def handle_user_turn(user_message: str) -> str:
    language = st.session_state.language
    user_in_english = translate_to_english(user_message, language)
    st.session_state.conversation.append({"role": "user", "content": user_in_english})
    assistant_response = run_conversation()
    translated_response = translate_from_english(assistant_response, language)
    st.session_state.messages.append({"role": "assistant", "content": translated_response})
    return translated_response


def get_available_devices():
    try:
        import pyaudio

        pa = pyaudio.PyAudio()
        devices = []
        for index in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(index)
            if info.get("maxInputChannels", 0) > 0:
                devices.append((index, info.get("name", f"Device {index}")))
        pa.terminate()
        return devices
    except ImportError:
        st.info("Install PyAudio to enable voice capture: pip install pipwin && pipwin install pyaudio")
        return []
    except Exception as exc:
        st.error(f"Could not access audio devices: {exc}")
        return []


def get_voice_query(device_index: Optional[int]) -> str:
    recogniser = sr.Recognizer()
    try:
        with sr.Microphone(device_index=device_index) as source:
            recogniser.adjust_for_ambient_noise(source, duration=1)
            audio = recogniser.listen(source, timeout=5, phrase_time_limit=10)
        return recogniser.recognize_google(audio)
    except sr.WaitTimeoutError:
        return "No speech detected."
    except Exception as exc:
        return f"Error: {exc}"


def process_image_query(image_path: str, user_query: str) -> str:
    if not GOOGLE_API_KEY:
        return "Image analysis requires a valid Google API key."
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    ]
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        generation_config=GENERATION_CONFIG,
        safety_settings=safety_settings
    )
    
    try:
        uploaded = genai.upload_file(path=image_path, display_name="fitness-image")
        
        analysis_prompt = (
            f"Fitness coach analyzing form. User asks: {user_query or 'Analyze this'}\n\n"
            "Respond in 3 bullet points:\n"
            "1. Exercise/equipment observed\n"
            "2. Form assessment\n"
            "3. One improvement tip\n"
            "Brief and actionable only."
        )
        
        response = model.generate_content([uploaded, analysis_prompt])
        text = extract_model_text(response)
        if text:
            return text
            
        reason = _describe_finish_reason(response)
        if reason:
            return (
                f"I couldn't analyze this image (safety filter: {reason}). "
                "Please ensure the image clearly shows exercise form or fitness equipment in a training context. "
                "Avoid images with sensitive content or unclear fitness context."
            )
        return "I couldn't interpret that image response. Please try again with a different prompt."
    except Exception as exc:
        error_msg = str(exc)
        if "block" in error_msg.lower() or "safety" in error_msg.lower():
            return (
                "This image was blocked by safety filters. Please upload a clear photo of:\n"
                "- Exercise form or technique\n"
                "- Workout equipment\n"
                "- Training environment\n"
                "Ensure the image is appropriate for fitness coaching context."
            )
        return f"Image analysis failed: {error_msg}"


def init_session_state():
    if "language" not in st.session_state:
        st.session_state.language = "en"
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "messages" not in st.session_state:
        welcome = "Ready with workouts, meals, and supplements. What's your goal?"
        st.session_state.messages = [{"role": "assistant", "content": welcome}]
        st.session_state.conversation.append({"role": "assistant", "content": welcome})


st.set_page_config(page_title="Recommendation Assistant", page_icon="💪", layout="wide")
init_session_state()


st.sidebar.markdown("### Assistant Controls")
st.session_state.language = st.sidebar.selectbox(
    "Response language",
    options=["en", "es", "fr", "de", "hi"],
    index=["en", "es", "fr", "de", "hi"].index(st.session_state.language)
    if st.session_state.language in ["en", "es", "fr", "de", "hi"] else 0,
)

st.sidebar.markdown(
    "**Dataset snapshot:** curated 7-day workout splits, macro-balanced meal templates, and vetted supplement summaries."
)
st.sidebar.markdown(
    "**Quick tips:** mention your fitness level, ask for meal swaps, or drop a progress photo for form cues."
)

with st.sidebar.expander("Voice input", expanded=False):
    devices = get_available_devices()
    if devices:
        device_choice = st.selectbox("Microphone", options=devices, format_func=lambda item: item[1])
        if st.button("🎙️ Record", use_container_width=True):
            transcript = get_voice_query(device_choice[0])
            if transcript.startswith("Error"):
                st.error(transcript)
            elif transcript == "No speech detected.":
                st.warning(transcript)
            else:
                st.success(f"Captured: {transcript}")
                st.session_state.messages.append({"role": "user", "content": transcript})
                with st.chat_message("user"):
                    st.markdown(transcript)
                with st.chat_message("assistant"):
                    with st.spinner("Processing voice query..."):
                        reply = handle_user_turn(transcript)
                        st.markdown(reply)
                rerun_app()
    else:
        st.caption("Connect a microphone to enable recording.")


chat_column, tools_column = st.columns([2.3, 1.0])

with chat_column:
    st.markdown("### Recommendation Assistant")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(f"Ask anything about workouts, meals, or supplements ({st.session_state.language.upper()})")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Crafting your plan..."):
                assistant_reply = handle_user_turn(prompt)
                st.markdown(assistant_reply)
        rerun_app()


with tools_column:
    st.markdown("### Fitness Image Assistance")
    uploaded_file = st.file_uploader("Drop a fitness image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
        image_query = st.text_area("Ask about this image", placeholder="e.g. How's my squat depth?", height=90)
        if st.button("Analyse Image", use_container_width=True):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name
            with st.spinner("Reviewing form..."):
                vision_reply_en = process_image_query(temp_path, image_query)
            os.remove(temp_path)

            language = st.session_state.language
            user_prompt = f"Image query: {image_query}" if image_query else "Image shared"
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            st.session_state.conversation.append({"role": "user", "content": translate_to_english(user_prompt, language)})

            translated_reply = translate_from_english(vision_reply_en, language)
            st.session_state.messages.append({"role": "assistant", "content": translated_reply})
            st.session_state.conversation.append({"role": "assistant", "content": vision_reply_en})

            with st.chat_message("user"):
                st.markdown(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(translated_reply)

            rerun_app()
