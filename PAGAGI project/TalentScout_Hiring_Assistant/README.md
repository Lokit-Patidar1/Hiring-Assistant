## TalentScout - AI Hiring Assistant

TalentScout is a Streamlit-based AI chatbot that assists a fictional recruitment agency in performing initial candidate screening. It gathers candidate details, confirms tech stack, generates technical questions using Google Gemini, maintains conversational context, and gracefully ends the session upon request. Optional sentiment analysis and language selection are included.

### Features
- Collects candidate info: Name, Email, Phone, Experience, Position, Location, Tech Stack
- Confirms candidate tech stack
- Generates 3–5 technical questions using Gemini based on tech stack
- Maintains conversation context and handles unclear inputs with a fallback
- Ends politely on “bye / thanks / exit” while summarizing collected info
- Bonus: Language toggle (English/Hindi) and sentiment analysis (TextBlob)

---

## Project Structure

```
TalentScout_Hiring_Assistant/
│
├── app.py               # Main Streamlit app
├── prompts.py           # System prompts and templates
├── utils.py             # Helper functions (Gemini, storage, sentiment)
├── requirements.txt     # Dependencies
├── README.md            # Documentation
└── data/
    └── candidates.csv   # Simulated data storage
```

---

## Setup Instructions

1) Clone or copy this folder to your local machine.

2) Create and activate a Python 3.10+ environment.

3) Install dependencies:

```bash
pip install -r requirements.txt
```

4) Provide your Google Gemini API key (`GOOGLE_API_KEY`):

- Option A (Environment Variable):

```bash
set GOOGLE_API_KEY=your_api_key_here  # Windows PowerShell: $env:GOOGLE_API_KEY="your_api_key_here"
```

- Option B (Streamlit Secrets): Create `.streamlit/secrets.toml` alongside your app run path with:

```toml
GOOGLE_API_KEY = "your_api_key_here"
```

---

## Running the App

From the project root (where `TalentScout_Hiring_Assistant` folder resides), run:

```bash
streamlit run TalentScout_Hiring_Assistant/app.py
```

Open the displayed local URL in your browser.

---

## Model & Library Details

- LLM: Google Gemini (default model: `gemini-1.5-flash`), used via `google.generativeai` Python package
- UI: Streamlit
- Storage: CSV via Pandas
- Sentiment: TextBlob (optional)

Gemini initialization (in code):

```python
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("your prompt here")
```

---

## Prompt Design

- Information Gathering Prompt (`INFO_GATHERING_INSTRUCTIONS`): Guides the model to ask for missing candidate details in a friendly, concise way, with short next-step questions. Language preference (English/Hindi) is passed to encourage responses in the chosen language.

- Technical Question Prompt (`QUESTION_GENERATION_TEMPLATE`): Requests 3–5 moderate questions focused on core concepts and problem-solving for the provided tech stack. The app parses the response into a clean list.

---

## Challenges & Solutions

- Conversation Flow Control: Combined lightweight deterministic parsing (for email/phone/experience) with Gemini-driven phrasing to keep the conversation natural while reliably storing structured data.

- Robustness of Question Lists: Some LLM outputs vary; added parsing and fallback logic to ensure 3–5 usable questions are shown.

- Graceful Ending: Detects common goodbye/thanks keywords, summarizes data, saves to CSV, and closes the session.

---

## Notes
- This project is intended for demo purposes only and does not include authentication or production-grade data storage.
- Ensure your API usage complies with the terms of the Gemini API.


