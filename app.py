import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer, MarianMTModel, MarianTokenizer
import fitz  # PyMuPDF
import os
import re
from langdetect import detect
import easyocr
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(page_title="Multilingual Text Summarizer", page_icon="📝", layout="wide")

@st.cache_resource
def load_model():
    model_directory = "t5-base"  # Using T5 for multilingual support
    model = T5ForConditionalGeneration.from_pretrained(model_directory)
    tokenizer = T5Tokenizer.from_pretrained(model_directory)
    return model, tokenizer

model, tokenizer = load_model()

@st.cache_resource
def load_translation_models():
    # Load translation models
    translation_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    translation_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-mul-en")
    return translation_model, translation_tokenizer

translation_model, translation_tokenizer = load_translation_models()

def translate_text(text, src_lang):
    # Translate text to English
    src_lang = src_lang.lower()
    if src_lang == "zh-cn":
        src_lang = "zh"
    translation_input = translation_tokenizer.prepare_seq2seq_batch([text], src_lang=src_lang, tgt_lang="en", return_tensors="pt")
    translated_ids = translation_model.generate(**translation_input)
    translated_text = translation_tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text

def preprocess_text(text):
    # Remove special characters and extra whitespace
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    cleaned_text = cleaned_text.strip()
    return cleaned_text

def summarize_text(text, prompts):
    cleaned_text = preprocess_text(text)
    combined_text = f"summarize: {cleaned_text}"
    if prompts:
        combined_text += " " + " ".join(prompts)
    
    tokenized_text = tokenizer.encode(combined_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    summary_ids = model.generate(tokenized_text, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

def read_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

def read_txt(file):
    return file.read().decode("utf-8")

def read_image(file, lang):
    image = Image.open(file)
    image_np = np.array(image)  # Convert PIL Image to numpy array
    
    # Language groups
    latin_languages = ['en', 'fr', 'de', 'es', 'it', 'pt']
    cyrillic_languages = ['ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'en']
    ja_ko_zh_languages = ['ja', 'ko', 'zh-cn', 'zh-tw', 'en']
    
    if lang in ['ja', 'ko', 'zh-cn', 'zh-tw']:
        reader = easyocr.Reader(ja_ko_zh_languages)
    elif lang in cyrillic_languages:
        reader = easyocr.Reader(cyrillic_languages)
    else:
        reader = easyocr.Reader(latin_languages)
    
    result = reader.readtext(image_np, detail=0)
    
    text = ' '.join(result)
    return text

def detect_language(text):
    lang = detect(text)
    return lang

# App layout
st.title("📝 Multilingual Text Summarizer")
st.write("**Welcome to the multilingual text summarizer!** Enter your text directly or upload a text/PDF/image file below, and let's create a concise summary together. 🧠")
st.write("**Using Transformer Model: T5**")

# Sidebar input method selection
st.sidebar.write("### Input Method")
input_method = st.sidebar.radio("Choose input method:", ("Direct Text Input", "Upload File (PDF, TXT, Image)"))

if input_method == "Direct Text Input":
    # Text input
    user_input = st.text_area("Enter your text here:", height=200)

    if user_input:
        file_text = user_input
    else:
        file_text = None

else:
    # File upload
    uploaded_file = st.file_uploader("Choose a file (PDF, TXT, Image)", type=["pdf", "txt", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == ".pdf":
            file_text = read_pdf(uploaded_file)
        elif file_extension == ".txt":
            file_text = read_txt(uploaded_file)
        elif file_extension in [".png", ".jpg", ".jpeg"]:
            # First detect the language of the image text
            temp_image_text = read_image(uploaded_file, 'en')  # Use English as a placeholder for detection
            detected_lang = detect_language(temp_image_text)
            file_text = read_image(uploaded_file, detected_lang)
        else:
            file_text = None
            st.error("Unsupported file type. Please upload a PDF, TXT, or Image file.")
    else:
        file_text = None

if file_text:
    if input_method == "Upload File (PDF, TXT, Image)":
        st.write("**File/Text content:**")
        st.text_area("File/Text content", value=file_text, height=200)

    # Detect language
    detected_language = detect_language(file_text)
    st.write(f"**Detected Language:** {detected_language.capitalize()}")

    # Translation option
    if detected_language != "en":
        translate_option = st.checkbox("Translate to English")
        if translate_option:
            file_text = translate_text(file_text, detected_language)
            st.write("**Translated Text:**")
            st.text_area("Translated Text", value=file_text, height=200)
            detected_language = "en"

    # Chat-like prompt system
    if "prompts" not in st.session_state:
        st.session_state.prompts = []

    st.write("### Refine your summary:")
    prompt = st.text_input("Enter a prompt to refine the summary, e.g., 'focus on key points'")

    if st.button("Add Prompt"):
        if prompt:
            st.session_state.prompts.append(prompt)
            st.success(f"Prompt added: {prompt}")
        else:
            st.error("Please enter a valid prompt.")

    # Display current prompts
    if st.session_state.prompts:
        st.write("#### Current Prompts:")
        for i, p in enumerate(st.session_state.prompts):
            st.write(f"{i+1}. {p}")

    # Summary button
    if st.button("Generate Summary"):
        with st.spinner("Generating summary..."):
            try:
                summary = summarize_text(file_text, st.session_state.prompts)
                st.subheader("Summary")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.write("Please enter some text or upload a file to get started.")

# CSS for styling
st.markdown("""
    <style>
    .stTextArea, .stTextInput, .stButton, .stMarkdown {
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)