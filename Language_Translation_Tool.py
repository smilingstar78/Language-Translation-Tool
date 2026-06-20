import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load API key
load_dotenv()

# ======================
# MODEL SETUP
# ======================
model = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

# ======================
# PROMPT (AUTO DETECT SOURCE LANGUAGE)
# ======================
prompt = PromptTemplate(
    template="""
You are a professional translation engine.

Step 1: Detect the language of the input text.
Step 2: Translate it into the target language.

Rules:
- Return ONLY translated text.
- Do NOT explain anything.

Input Text: {text}
Target Language: {target_lang}
""",
    input_variables=["text", "target_lang"]
)

parser = StrOutputParser()

chain = prompt | model | parser


# ======================
# TRANSLATION FUNCTION
# ======================
def translate(text, target_lang):
    return chain.invoke({
        "text": text,
        "target_lang": target_lang
    })


# ======================
# STREAMLIT UI
# ======================
st.set_page_config(page_title="AI Translator", layout="wide")

st.title("AI Language Translation Tool")
st.write("Paste text on the left and get translation on the right ⚡")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Text")
    input_text = st.text_area("Enter text here", height=100)

with col2:
    st.subheader("Output")

    target_lang = st.selectbox(
        "Select target language",
        [
            "English", "Hindi","Urdu", "Arabic", "Spanish",
            "French", "German", "Chinese", "Japanese", "Turkish"
        ]
    )

    if st.button("Translate"):
        if input_text.strip() == "":
            st.warning("Please enter some text first!")
        else:
            with st.spinner("Translating..."):
                result = translate(input_text, target_lang)
                st.success(result)