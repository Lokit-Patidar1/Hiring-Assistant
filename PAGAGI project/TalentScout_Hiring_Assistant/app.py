import os
import re
from typing import Dict, List

import streamlit as st

from prompts import INFO_GATHERING_INSTRUCTIONS, UNKNOWN_FALLBACK
from utils import (
	init_gemini_model,
	generate_questions,
	save_candidate_row,
	is_goodbye,
	get_missing_fields,
	parse_tech_stack,
	blob_sentiment,
	random_goodbye,
	resolve_supported_model,
)

APP_TITLE = "TalentScout - AI Hiring Assistant"
DATA_PATH = os.path.join("TalentScout_Hiring_Assistant", "data", "candidates.csv")
MODEL_NAME = "gemini-1.5-flash-latest"


def _init_state():
	if "messages" not in st.session_state:
		st.session_state.messages = []  # List[Dict[str, str]] with keys: role ("assistant"|"user"), "content"
	if "candidate" not in st.session_state:
		st.session_state.candidate = {
			"name": "",
			"email": "",
			"phone": "",
			"experience": "",
			"position": "",
			"location": "",
			"tech_stack": "",
		}
	if "tech_list" not in st.session_state:
		st.session_state.tech_list: List[str] = []
	if "model" not in st.session_state:
		st.session_state.model = None
	if "model_name" not in st.session_state:
		st.session_state.model_name = MODEL_NAME
	if "current_field" not in st.session_state:
		st.session_state.current_field = ""
	if "language" not in st.session_state:
		st.session_state.language = "English"
	if "session_ended" not in st.session_state:
		st.session_state.session_ended = False
	if "asked_questions" not in st.session_state:
		st.session_state.asked_questions: List[str] = []
	if "info_collection_complete" not in st.session_state:
		st.session_state.info_collection_complete = False
	if "chat_mode" not in st.session_state:
		st.session_state.chat_mode = False
	if "last_processed_input" not in st.session_state:
		st.session_state.last_processed_input = ""


def _language_texts(lang: str) -> Dict[str, str]:
	if lang == "Hindi":
		return {
			"greeting": "नमस्ते! मैं TalentScout हूँ, आपका AI Hiring Assistant। मैं आपकी प्रारंभिक स्क्रीनिंग के लिए कुछ विवरण एकत्र करूँगा। क्या आप तैयार हैं?",
			"ask_name": "कृपया अपना पूरा नाम बताएं।",
			"ask_email": "कृपया अपना ईमेल बताएं।",
			"ask_phone": "कृपया अपना फ़ोन नंबर बताएं।",
			"ask_experience": "आपके पास कुल कितने वर्षों का अनुभव है?",
			"ask_position": "आप किस पद के लिए आवेदन कर रहे हैं?",
			"ask_location": "आपका वर्तमान स्थान क्या है?",
			"ask_tech": "कृपया अपना टेक स्टैक बताएं (जैसे Python, Django, React)।",
			"confirm_tech": "धन्यवाद! आपने ये तकनीकें बताई हैं: ",
			"questions_intro": "यहाँ आपकी तकनीकों के आधार पर कुछ तकनीकी प्रश्न हैं:",
			"fallback": "क्षमा करें, क्या आप कृपया स्पष्ट कर सकते हैं?",
			"ended": "बातचीत समाप्त हो गई है। धन्यवाद!",
			"summary_title": "आपकी जानकारी का सारांश",
			"sentiment": "आपका मूड अनुमान",
		}
	return {
		"greeting": "Hello! I’m TalentScout, your AI Hiring Assistant. I’ll collect a few details for initial screening. Shall we begin?",
		"ask_name": "Please share your full name.",
		"ask_email": "Please share your email address.",
		"ask_phone": "Please share your phone number.",
		"ask_experience": "How many total years of experience do you have?",
		"ask_position": "What position are you applying for?",
		"ask_location": "What is your current location?",
		"ask_tech": "Please list your tech stack (e.g., Python, Django, React).",
		"confirm_tech": "Thanks! You’ve listed the following technologies: ",
		"questions_intro": "Here are a few technical questions based on your tech stack:",
		"fallback": "I’m sorry, could you please clarify that?",
		"ended": "The session has ended. Thank you!",
		"summary_title": "Summary of your details",
		"sentiment": "Your mood estimate",
	}


def _field_label(field_key: str, lang_texts: Dict[str, str]) -> str:
	return {
		"name": lang_texts["ask_name"],
		"email": lang_texts["ask_email"],
		"phone": lang_texts["ask_phone"],
		"experience": lang_texts["ask_experience"],
		"position": lang_texts["ask_position"],
		"location": lang_texts["ask_location"],
		"tech_stack": lang_texts["ask_tech"],
	}.get(field_key, lang_texts["fallback"])


def _extract_value_for_field(user_text: str, field: str) -> str:
	text = user_text.strip()
	if field == "email":
		# Basic heuristic
		match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
		return match.group(0) if match else text
	if field == "phone":
		digits = re.sub(r"\D", "", text)
		return digits if len(digits) >= 7 else text
	if field == "experience":
		match = re.search(r"(\d+(\.\d+)?)", text)
		return match.group(1) if match else text
	if field == "tech_stack":
		# Store as comma string; keep list separately too
		techs = parse_tech_stack(text)
		return ", ".join(techs)
	return text


def _llm_next_prompt(model, candidate: Dict[str, str], language: str) -> str:
	known_lines = []
	for k, v in candidate.items():
		if v:
			known_lines.append(f"{k}: {v}")
	known_block = "\n".join(known_lines) if known_lines else "None yet"

	lang_hint = f"Respond in {language}." if language else ""
	prompt = (
		f"{INFO_GATHERING_INSTRUCTIONS}\n\n"
		f"Candidate info known so far:\n{known_block}\n\n"
		f"{lang_hint}\n"
		f"Kindly provide a short, friendly next question to collect the missing items."
	)
	try:
		resp = model.generate_content(prompt)
		text = (resp.text or "").strip()
		return text if text else UNKNOWN_FALLBACK
	except Exception:
		return UNKNOWN_FALLBACK


def main():
	st.set_page_config(page_title=APP_TITLE, page_icon="🤖", layout="centered")
	_init_state()

	# Sidebar controls
	with st.sidebar:
		st.header("TalentScout")
		st.session_state.language = st.selectbox("Language", ["English", "Hindi"], index=0)

		# Model selection (includes 2.5 flash option)
		model_options = [
			"gemini-2.5-flash",
			"gemini-2.0-flash",
			"gemini-1.5-flash-latest",
			"gemini-1.5-flash-8b",
			"gemini-1.5-pro",
			"gemini-1.5-pro-latest",
		]
		if st.session_state.model_name not in model_options:
			model_options = [st.session_state.model_name] + model_options
		st.session_state.model_name = st.selectbox("Model", model_options, index=model_options.index(st.session_state.model_name) if st.session_state.model_name in model_options else 0)

		api_key_from_secrets = st.secrets.get("GOOGLE_API_KEY", None) if hasattr(st, "secrets") else None
		api_key = api_key_from_secrets or os.getenv("GOOGLE_API_KEY") or ""
		st.caption("Using GOOGLE_API_KEY from secrets or environment.")

		if st.button("Reset Conversation", type="secondary"):
			for k in ["messages", "candidate", "tech_list", "current_field", "session_ended", "asked_questions", "info_collection_complete", "chat_mode", "last_processed_input"]:
				if k == "candidate":
					st.session_state[k] = {"name": "", "email": "", "phone": "", "experience": "", "position": "", "location": "", "tech_stack": ""}
				else:
					st.session_state[k] = [] if isinstance(st.session_state.get(k), list) else False if isinstance(st.session_state.get(k), bool) else ""
			st.session_state.session_ended = False
			st.session_state.info_collection_complete = False
			st.session_state.chat_mode = False
			# Do not reset selected model/language to preserve user choices
			st.rerun()

		st.divider()
		st.subheader("Collected Info")
		for key, val in st.session_state.candidate.items():
			st.write(f"- {key}: {val or '—'}")

	# Model init
	if st.session_state.model is None:
		try:
			resolved = resolve_supported_model(st.session_state.model_name)
			st.session_state.model = init_gemini_model(resolved, api_key=api_key)
			st.session_state.initialized_model_name = resolved
		except Exception as e:
			st.error(str(e))
			st.stop()
	else:
		# Reinitialize model if user changed selection
		if getattr(st.session_state, "initialized_model_name", None) != st.session_state.model_name:
			try:
				resolved = resolve_supported_model(st.session_state.model_name)
				st.session_state.model = init_gemini_model(resolved, api_key=api_key)
				st.session_state.initialized_model_name = resolved
			except Exception as e:
				st.error(str(e))
				st.stop()

	lang_texts = _language_texts(st.session_state.language)

	# Initial greeting
	if not st.session_state.messages:
		st.session_state.messages.append({"role": "assistant", "content": lang_texts["greeting"]})
		# Ask first missing field immediately to avoid multiple questions in one line
		start_missing = get_missing_fields(st.session_state.candidate)
		if start_missing:
			st.session_state.current_field = start_missing[0]
			first_q = _field_label(st.session_state.current_field, lang_texts)
			with st.chat_message("assistant"):
				st.write(first_q)
			st.session_state.messages.append({"role": "assistant", "content": first_q})

	# Chat display
	for msg in st.session_state.messages:
		with st.chat_message(msg["role"]):
			st.write(msg["content"])

	# Input disabled after session end
	placeholder = "Type your message..." if st.session_state.language == "English" else "अपना संदेश लिखें..."
	user_input = st.chat_input(placeholder=placeholder, disabled=st.session_state.session_ended)

	# Skip processing if this input was already processed
	if user_input and user_input == st.session_state.last_processed_input:
		user_input = None

	if user_input:
		# Sentiment (optional)
		polarity, mood = blob_sentiment(user_input)
		mood_display = {
			"positive": "😊",
			"neutral": "😐",
			"negative": "🙁",
		}.get(mood, "😐")

		st.session_state.messages.append({"role": "user", "content": user_input})
		st.session_state.last_processed_input = user_input

		# End sequence
		if is_goodbye(user_input):
			summary_lines = [f"{k}: {v}" for k, v in st.session_state.candidate.items() if v]
			summary = "\n".join(summary_lines) or "No details provided."
			with st.chat_message("assistant"):
				st.write(random_goodbye())
				st.write(f"({lang_texts['sentiment']}: {mood} {mood_display}, polarity={polarity:.2f})")
				st.markdown(f"**{lang_texts['summary_title']}:**\n\n{summary}")
			st.session_state.messages.append({"role": "assistant", "content": random_goodbye()})
			# Save row
			try:
				save_candidate_row(DATA_PATH, st.session_state.candidate)
			except Exception as e:
				st.warning(f"Could not save data: {e}")
			st.session_state.session_ended = True
			st.stop()

		# Collect information
		cand = st.session_state.candidate
		# Always capture the currently asked field only (one-question-per-turn)
		# Update immediately when field is provided
		info_updated = False
		if st.session_state.current_field:
			field = st.session_state.current_field
			old_value = cand.get(field, "")
			value = _extract_value_for_field(user_input, field)
			cand[field] = value
			if field == "tech_stack":
				st.session_state.tech_list = parse_tech_stack(value)
			# Mark that info was updated
			if old_value != value:
				info_updated = True

		# Determine next action
		missing_after = get_missing_fields(cand)
		
		# Check if we're in chat mode (after info collection and questions)
		if st.session_state.info_collection_complete and st.session_state.chat_mode:
			# Handle general questions using LLM
			chat_prompt = f"""You are TalentScout, a helpful AI hiring assistant. The candidate has completed their initial screening. 
They have provided: {', '.join([f"{k}: {v}" for k, v in cand.items() if v])}
Their tech stack: {', '.join(st.session_state.tech_list) if st.session_state.tech_list else 'Not specified'}

The candidate just asked: "{user_input}"

Provide a helpful, concise response. If they're asking about the interview process, their application, or technical questions, be informative and friendly.
Keep responses brief (2-3 sentences max). Respond in {st.session_state.language}."""
			
			with st.chat_message("assistant"):
				with st.spinner("Thinking..."):
					try:
						response = st.session_state.model.generate_content(chat_prompt)
						assistant_response = (response.text or "").strip()
						if not assistant_response:
							assistant_response = UNKNOWN_FALLBACK
						st.write(assistant_response)
						st.session_state.messages.append({"role": "assistant", "content": assistant_response})
					except Exception as e:
						error_msg = f"I apologize, I encountered an error. Please try again. ({str(e)})"
						st.write(error_msg)
						st.session_state.messages.append({"role": "assistant", "content": error_msg})
		elif not missing_after:
			# All info collected - generate questions if not already done
			if st.session_state.tech_list and not st.session_state.asked_questions:
				confirm_msg = f"{_language_texts(st.session_state.language)['confirm_tech']}" + ", ".join(st.session_state.tech_list)
				with st.chat_message("assistant"):
					st.write(confirm_msg)
					st.write(lang_texts["questions_intro"])
					with st.spinner("Generating questions..."):
						qs = generate_questions(st.session_state.tech_list, st.session_state.model)
					if qs:
						for q in qs:
							st.write(f"- {q}")
						st.session_state.asked_questions = qs
					else:
						st.write(UNKNOWN_FALLBACK)

				st.session_state.messages.append({"role": "assistant", "content": confirm_msg})
				st.session_state.info_collection_complete = True
				st.session_state.chat_mode = True

			# Enter chat mode after questions are shown
			if st.session_state.asked_questions and not st.session_state.chat_mode:
				st.session_state.info_collection_complete = True
				st.session_state.chat_mode = True
				# Provide graceful next step when entering chat mode
				next_prompt = "Great! I'm here to help. Feel free to ask me any questions about the interview process, your application, or anything else. If you'd like to end, say 'bye'."
				if st.session_state.language == "Hindi":
					next_prompt = "बढ़िया! मैं यहाँ आपकी मदद के लिए हूँ। साक्षात्कार प्रक्रिया, आपके आवेदन, या किसी अन्य चीज़ के बारे में कोई भी प्रश्न पूछने के लिए स्वतंत्र महसूस करें। यदि आप समाप्त करना चाहते हैं, तो 'bye' कहें।"
				with st.chat_message("assistant"):
					st.write(next_prompt)
				st.session_state.messages.append({"role": "assistant", "content": next_prompt})
			st.session_state.current_field = ""
		else:
			# Ask the next single question deterministically to reduce lag (no LLM call)
			st.session_state.current_field = missing_after[0]
			next_q = _field_label(st.session_state.current_field, lang_texts)
			with st.chat_message("assistant"):
				st.write(next_q)
			st.session_state.messages.append({"role": "assistant", "content": next_q})
		
		# If info was updated, trigger rerun at the end to update sidebar immediately
		# This happens after all processing is complete
		if info_updated:
			st.rerun()


if __name__ == "__main__":
	main()


