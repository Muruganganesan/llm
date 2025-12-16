import google.generativeai as genai
import os

# 🔑 API key (better: environment variable)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

#st.secrets["GEMINI_API_KEY"]

model = genai.GenerativeModel("gemini-1.5-flash")

def summarize(text, max_words):
    prompt = f"""
    Summarize the following text in about {max_words} words.
    Be clear, concise, and avoid repetition.

    Text:
    {text}
    """

    response = model.generate_content(prompt)
    return response.text
