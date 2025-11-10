import os
import random
from typing import Dict, List, Tuple

import pandas as pd

try:
	import google.generativeai as genai
except Exception:  # pragma: no cover
	genai = None

try:
	from textblob import TextBlob
except Exception:  # pragma: no cover
	TextBlob = None


REQUIRED_FIELDS = ["name", "email", "phone", "experience", "position", "location", "tech_stack"]


def get_env_api_key() -> str:
	# Prefer Streamlit secrets if available (runtime); fallback to environment
	api_key = os.getenv("GOOGLE_API_KEY")
	if not api_key:
		# Streamlit secrets access is dynamic inside Streamlit, handled in app.py
		return ""
	return api_key


def init_gemini_model(model_name: str = "gemini-1.5-flash-latest", api_key: str = ""):
	# Lazy init to avoid import issues when library not installed
	if genai is None:
		raise RuntimeError("google-generativeai is not installed. Please install dependencies from requirements.txt")

	if not api_key:
		api_key = get_env_api_key()

	if not api_key:
		raise RuntimeError("Missing GOOGLE_API_KEY. Set it in environment or Streamlit secrets.")

	genai.configure(api_key=api_key)
	return genai.GenerativeModel(model_name)


def resolve_supported_model(preferred_name: str) -> str:
	"""
	Resolve a supported model name for the current SDK/account by listing models and
	picking one that supports generateContent. Returns full model name (may be 'models/...').
	"""
	if genai is None:
		return preferred_name
	try:
		models = list(genai.list_models())
		def supports_generate(m) -> bool:
			methods = getattr(m, "supported_generation_methods", None) or []
			return "generateContent" in methods

		def name_matches(m, short_or_full: str) -> bool:
			return m.name == short_or_full or m.name.endswith("/" + short_or_full)

		supported = [m for m in models if supports_generate(m)]
		# Exact preferred match
		for m in supported:
			if name_matches(m, preferred_name):
				return m.name

		# Preference fallbacks
		preferences = [
			preferred_name,
			"gemini-2.5-flash",
			"gemini-2.0-flash",
			"gemini-1.5-flash",
			"gemini-1.5-pro",
		]
		for cand in preferences:
			for m in supported:
				if name_matches(m, cand):
					return m.name
		# Last resort
		if supported:
			return supported[0].name
	except Exception:
		return preferred_name
	return preferred_name


def generate_questions(tech_stack_list: List[str], model) -> List[str]:
	if not tech_stack_list:
		return []

	from prompts import QUESTION_GENERATION_TEMPLATE
	prompt = QUESTION_GENERATION_TEMPLATE.format(tech_stack_list=", ".join(tech_stack_list))
	response = model.generate_content(prompt)
	text = (response.text or "").strip()
	if not text:
		return []

	# Split into lines; extract numbered list items
	lines = [line.strip() for line in text.splitlines() if line.strip()]
	questions: List[str] = []
	for line in lines:
		# Remove leading numbering like "1. " or "- "
		clean = line
		if clean[:2].isdigit() and clean[1:2] == ".":
			clean = clean[2:].strip()
		if clean[:3].isdigit() and clean[2:3] == ".":
			clean = clean[3:].strip()
		if clean.startswith(("-", "*")):
			clean = clean[1:].strip()
		if len(clean) > 0:
			questions.append(clean)

	# Keep 3-5
	if len(questions) > 5:
		return questions[:5]
	if len(questions) < 3 and len(lines) >= 3:
		# Fallback: return first 3 raw lines
		return lines[:3]
	return questions


def save_candidate_row(filepath: str, candidate: Dict[str, str]) -> None:
	os.makedirs(os.path.dirname(filepath), exist_ok=True)
	columns = ["name", "email", "phone", "experience", "position", "location", "tech_stack"]
	row = {col: candidate.get(col, "") for col in columns}

	# Create file if not exists
	if not os.path.exists(filepath):
		df = pd.DataFrame(columns=columns)
		df.to_csv(filepath, index=False)

	df = pd.read_csv(filepath)
	df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
	df.to_csv(filepath, index=False)


def is_goodbye(text: str) -> bool:
	if not text:
		return False
	lower = text.strip().lower()
	keywords = ["bye", "goodbye", "thank you", "thanks", "exit", "quit", "see you"]
	return any(k in lower for k in keywords)


def get_missing_fields(candidate: Dict[str, str]) -> List[str]:
	return [f for f in REQUIRED_FIELDS if not candidate.get(f)]


def parse_tech_stack(value: str) -> List[str]:
	if not value:
		return []
	parts = [p.strip() for p in value.replace(" and ", ",").split(",")]
	return [p for p in parts if p]


def blob_sentiment(text: str) -> Tuple[float, str]:
	if TextBlob is None or not text:
		return 0.0, "neutral"
	polarity = float(TextBlob(text).sentiment.polarity)
	if polarity > 0.2:
		return polarity, "positive"
	if polarity < -0.2:
		return polarity, "negative"
	return polarity, "neutral"


def random_goodbye() -> str:
	from prompts import GOODBYE_RESPONSES
	return random.choice(GOODBYE_RESPONSES)


