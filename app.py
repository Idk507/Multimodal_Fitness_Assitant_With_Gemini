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
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 1024,
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

FITNESS_SYSTEM_PROMPT = (
    "You are a supportive fitness assistant. Answer the user's request directly and keep responses concise."
)

ACTION_GUIDE = (
    "If the user asks for a workout, meal plan, motivational quote, protein recommendation, or diet-friendly places,"
    " call the matching tool using one line formatted exactly as: ACTION: <tool_name> | <input>."
    " Available tools: generate_workout, suggest_meal, motivational_quotes, recommend_protein, recommend_diet_hotels."
    " After receiving tool results, give a final helpful answer without requesting another action."
)

def _normalise_level(level: str) -> str:
    token = (level or "").strip().lower()
    if "beg" in token:
        return "beginner"
    if "inter" in token:
        return "intermediate"
    if "adv" in token:
        return "advanced"
    return "beginner"

def _looks_unusable(text: str) -> bool:
    if not text:
        return True
    lowered = text.lower()
    failure_markers = [
        "snag",
        "try again",
        "api key",
        "error",
        "unavailable",
    ]
    return any(marker in lowered for marker in failure_markers)


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


def generate_with_retry(prompt_builder, max_attempts: int = 3) -> str:
    last_response = ""
    for attempt in range(max_attempts):
        prompt = prompt_builder(attempt)
        response = generate_text(prompt)
        if not _looks_unusable(response):
            return response
        last_response = response
    return (
        "I'm running into upstream limits right now, so I can't share the full plan yet. "
        f"(Latest response: {last_response})"
    )


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
    for message in conversation:
        role = message["role"]
        content = message["content"].strip()
        if not content:
            continue
        if role == "user":
            history_lines.append(f"User: {content}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {content}")
        elif role == "tool":
            history_lines.append(f"Tool: {content}")

    history_text = "\n".join(history_lines).strip()
    guidance = FITNESS_SYSTEM_PROMPT
    if allow_action:
        guidance = f"{FITNESS_SYSTEM_PROMPT}\n\n{ACTION_GUIDE}"
    else:
        guidance = (
            f"{FITNESS_SYSTEM_PROMPT}\n\nYou already have the tool output above. Summarise it and answer the user without calling any more tools."
        )

    prompt_parts = [guidance]
    if history_text:
        prompt_parts.append(f"Conversation so far:\n{history_text}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)


def generate_workout(level: str) -> str:
    normalised_level = _normalise_level(level)
    def build_prompt(attempt: int) -> str:
        reinforcement = "Respond in Markdown with a header and a table that has columns Day, Focus, Warm-up, Main Sets, Cool-down."
        if attempt > 0:
            reinforcement += " Avoid apologies or deferrals—produce the plan immediately."
        if attempt > 1:
            reinforcement += " Keep the table to 7 rows, one per day, and include short coaching cues in the Main Sets column."
        return (
            f"User fitness level: {normalised_level}. Create a 7-day workout schedule with balanced strength, cardio, and recovery.\n"
            f"{reinforcement}"
        )

    return generate_with_retry(build_prompt)


def suggest_meal(preferences: str) -> str:
    preferences = preferences or "high-protein balanced nutrition"
    prompt = (
        f"{FITNESS_SYSTEM_PROMPT}\n\n"
        f"Create a one-day meal outline aligned with {preferences}. List breakfast, lunch, dinner, and snacks with macros." 
    )
    return generate_text(prompt)


def motivational_quotes(_: str = "") -> str:
    prompt = (
        f"{FITNESS_SYSTEM_PROMPT}\n\nShare one short motivational quote tailored to someone working on their fitness routine." 
    )
    return generate_text(prompt)


def summarise_search(query: str, task_instruction: str) -> str:
    try:
        snippets = search_tool.run(query)
    except Exception:
        return "I could not reach the search service just now. Please try again soon."
    prompt = (
        f"{FITNESS_SYSTEM_PROMPT}\n\n"
        f"Search snippets for '{query}':\n{snippets}\n\n"
        f"Task: {task_instruction}"
    )
    return generate_text(prompt)


def recommend_protein(focus: str = "") -> str:
    query = "best live protein supplements with nutrition facts and prices"
    if focus:
        query += f" for {focus}"
    task = (
        "Summarise the top options with serving size, protein per scoop, key ingredients, price range, and purchase links when present. "
        "Conclude with how to pick the right supplement based on goals."
    )
    return summarise_search(query, task)


def recommend_diet_hotels(location: str = "") -> str:
    query = "best healthy meal prep services and diet-friendly restaurants online"
    if location:
        query += f" in {location}"
    task = (
        "Highlight 3-4 services with signature dishes, macro focus, ordering method, and delivery or dining details."
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
            "You are a professional fitness coach analyzing exercise form and technique. "
            "This is for legitimate fitness education purposes.\n\n"
            f"User question: {user_query}\n\n"
            "Provide constructive coaching feedback covering:\n"
            "1. What you observe in the image (exercise, setting, equipment)\n"
            "2. Form and technique observations\n"
            "3. Specific improvement suggestions\n"
            "Keep your response professional, educational, and focused on athletic performance."
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
        welcome = "Hey! I'm ready with workouts, meals, and supplement pointers. Tell me your goal to get started."
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
