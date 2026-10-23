from io import BytesIO
import html
import json

import streamlit as st
import torch
from PIL import Image
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoProcessor,
    AutoTokenizer,
    BlipForConditionalGeneration,
)

MODEL_NAME = "google/flan-t5-base"
CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
TRANSLATION_MODEL_NAME = "facebook/nllb-200-distilled-600M"
LANGUAGES = {
    "English": "eng_Latn",
    "Tamil": "tam_Taml",
    "Hindi": "hin_Deva",
}

st.set_page_config(
    page_title="AI News Headline Generator",
    page_icon="📰",
    layout="wide"
)

st.markdown("""
<style>
:root {
    --ink: #173b48;
    --muted: #526d77;
    --teal: #14756b;
    --teal-dark: #0b5145;
    --coral: #d76531;
    --line: #c8dfe2;
}
[data-testid="stAppViewContainer"] {
    background: linear-gradient(128deg, #bfeaff 0%, #d9f1a7 42%, #fff0a3 76%, #d4efd5 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] {
    background: linear-gradient(165deg, #124a46 0%, #176a58 55%, #397a49 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.25);
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
    color: #f3fffc;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
    border-color: rgba(255, 255, 255, 0.65);
    background: rgba(255, 255, 255, 0.96);
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {
    color: var(--ink);
}
.block-container { max-width: 1120px; padding-top: 2rem; padding-bottom: 3rem; }
h1, h2, h3 { color: var(--ink); letter-spacing: 0; }
.app-hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
    border: 1px solid var(--line);
    border-top: 4px solid var(--teal);
    border-radius: 10px;
    background: linear-gradient(110deg, #ffffff 0%, #e8f6ff 72%, #f4f8d9 100%);
    box-shadow: 0 10px 28px rgba(24, 57, 67, 0.07);
}
.app-kicker {
    color: var(--teal);
    font-size: 0.76rem;
    font-weight: 800;
    text-transform: uppercase;
}
.app-hero h1 { margin: 0.35rem 0 0; font-size: 2rem; }
.app-hero p { margin: 0.35rem 0 0; color: var(--muted); }
.hero-mark { font-size: 3rem; }
[data-testid="stTextArea"] textarea {
    border: 1px solid #c6dce1;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.92);
    transition: border-color 160ms ease, box-shadow 160ms ease;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--teal);
    box-shadow: 0 0 0 3px rgba(8, 127, 120, 0.14);
}
[data-testid="stFileUploader"] {
    padding: 0.6rem;
    border: 1px dashed #9dbbb7;
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.65);
}
[data-testid="stSelectbox"] > div > div {
    border-radius: 7px;
    border-color: #c6dce1;
    background: #fff;
}
[data-testid="stMetric"] {
    padding: 0.85rem 1rem;
    border: 1px solid var(--line);
    border-top: 3px solid var(--teal);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 5px 16px rgba(24, 57, 67, 0.05);
}
[data-testid="stMetricValue"] { color: var(--ink); }
[data-testid="stButton"] button {
    min-height: 2.8rem;
    border-radius: 8px;
    font-weight: 700;
    transition: transform 150ms ease, box-shadow 150ms ease, background 150ms ease;
}
[data-testid="stButton"] button[kind="primary"] {
    border: 0;
    color: white;
    background: linear-gradient(105deg, #14756b, #239b78);
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(105deg, #0b5145, #14756b);
    box-shadow: 0 7px 18px rgba(20, 117, 107, 0.24);
    transform: translateY(-1px);
}
[data-testid="stButton"] button[kind="secondary"]:hover {
    color: var(--coral);
    border-color: var(--coral);
    background: #fff8f4;
}
.headline-card {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    margin: 0.7rem 0 0.35rem;
    padding: 0.95rem 1rem;
    border: 1px solid var(--line);
    border-left: 4px solid var(--coral);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.92);
    color: var(--ink);
    font-size: 1.05rem;
    font-weight: 650;
    animation: rise-in 240ms ease-out both;
}
.headline-index {
    display: grid;
    flex: 0 0 2rem;
    width: 2rem;
    height: 2rem;
    place-items: center;
    border-radius: 6px;
    background: #e6f3d8;
    color: var(--teal-dark);
    font-size: 0.8rem;
    font-weight: 800;
}
@keyframes rise-in {
    from { opacity: 0; transform: translateY(5px); }
    to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 640px) {
    .block-container { padding: 1rem 1rem 2rem; }
    .app-hero { padding: 1.1rem; }
    .app-hero h1 { font-size: 1.55rem; }
    .hero-mark { font-size: 2.2rem; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


@st.cache_resource
def load_image_captioner():
    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
    return processor, model


@st.cache_resource
def load_translator():
    tokenizer = AutoTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(TRANSLATION_MODEL_NAME)
    return tokenizer, model


def caption_image(image_file):
    processor, model = load_image_captioner()
    image = Image.open(BytesIO(image_file.getvalue())).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40)

    return processor.decode(output[0], skip_special_tokens=True).strip()


def detect_article_language(article):
    if any("\u0B80" <= character <= "\u0BFF" for character in article):
        return "Tamil"
    if any("\u0900" <= character <= "\u097F" for character in article):
        return "Hindi"
    return "English"


def translate_texts(texts, source_language, target_language, max_new_tokens):
    if source_language == target_language:
        return texts

    tokenizer, model = load_translator()
    tokenizer.src_lang = source_language
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_language),
            max_new_tokens=max_new_tokens,
        )

    return [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]


def translate_headlines(headlines, target_language):
    return translate_texts(headlines, "eng_Latn", target_language, max_new_tokens=48)


def generate_headlines(article, tokenizer, model, image_caption=None, number=5):
    context = []
    if article.strip():
        context.append(f"Article:\n{article}")
    if image_caption:
        context.append(f"Image description:\n{image_caption}")

    prompt = (
        "Write a concise, factually grounded English news headline from this context. "
        "Use only facts stated here. Do not invent names, places, numbers, events, "
        "or outcomes. Focus on the central announcement. Return only the headline.\n\n"
        + "\n\n".join(context)
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=30,
            min_length=5,
            num_beams=8,
            num_return_sequences=number,
            do_sample=False,
            no_repeat_ngram_size=2
        )

    headlines = []
    for output in outputs:
        text = tokenizer.decode(output, skip_special_tokens=True).strip()
        if text and text not in headlines:
            headlines.append(text)

    return headlines[:number]

def copy_button(text):
    safe_js_text = html.escape(json.dumps(text, ensure_ascii=False), quote=True)
    component = f"""
    <div style="display:flex;align-items:center;gap:10px;margin:0 0 14px 3rem;">
      <button onclick="navigator.clipboard.writeText({safe_js_text}).then(()=>{{this.innerText='Copied';setTimeout(()=>this.innerText='Copy',1200)}})"
              style="padding:7px 12px;border:1px solid #c6ddd8;border-radius:7px;
                     background:#e9f5f2;color:#087f78;font-weight:700;cursor:pointer;">
        Copy
      </button>
    </div>
    """
    st.components.v1.html(component, height=42)

# ---------- UI ----------
if "form_version" not in st.session_state:
    st.session_state.form_version = 0

def clear_form():
    st.session_state.form_version += 1

form_version = st.session_state.form_version

st.markdown("""
<div class="app-hero">
  <div>
    <div class="app-kicker">Newsroom · NLP Studio</div>
    <h1>Automatic News Headline Generation</h1>
    <p>Turn a news story into a concise, publication-ready headline.</p>
  </div>
  <div class="hero-mark" aria-hidden="true">📰</div>
</div>
""", unsafe_allow_html=True)

st.subheader("Article input")
article = st.text_area(
    "Paste your article",
    height=220,
    placeholder="Paste a news article in English, Tamil, or Hindi...",
    key=f"article_{form_version}",
)

with st.sidebar:
    st.subheader("Headline settings")
    reference_image = st.file_uploader(
        "Reference image (optional)",
        type=["jpg", "jpeg", "png", "webp"],
        key=f"image_{form_version}",
    )
    if reference_image:
        st.image(reference_image, caption="Reference image", use_container_width=True)
    article_language_option = st.selectbox(
        "Article language",
        ["Auto-detect", *LANGUAGES],
        key=f"article_language_{form_version}",
    )
    selected_output_language = st.selectbox(
        "Headline language",
        ["Same as article", *LANGUAGES],
        key=f"output_language_{form_version}",
    )

article_language = (
    detect_article_language(article)
    if article_language_option == "Auto-detect"
    else article_language_option
)
output_language = (
    article_language
    if selected_output_language == "Same as article"
    else selected_output_language
)

word_count = len(article.split()) if article.strip() else 0
ready_to_generate = bool(article.strip() or reference_image) and (
    word_count >= 20 or bool(reference_image)
)
metric_cols = st.columns(3)
metric_cols[0].metric("Article words", word_count)
metric_cols[1].metric("Detected language", article_language)
metric_cols[2].metric("Headline language", output_language)
st.progress(1.0 if reference_image else min(word_count / 20, 1.0))
st.caption(
    "Ready to generate" if ready_to_generate else "Add a 20-word article or a reference image to continue."
)

col1, col2 = st.columns(2)

with col1:
    generate = st.button(
        "Generate headlines", type="primary", use_container_width=True,
        key=f"generate_{form_version}",
    )

with col2:
    st.button(
        "Clear", on_click=clear_form, use_container_width=True,
        key=f"clear_{form_version}",
    )

if generate:
    if not article.strip() and not reference_image:
        st.error("Please enter a news article or add a reference image.")
    elif article.strip() and word_count < 20 and not reference_image:
        st.warning("Please enter a longer article (at least 20 words) for better results.")
    else:
        with st.spinner("🤖 AI is analyzing the article and generating headlines..."):
            try:
                image_caption = caption_image(reference_image) if reference_image else None
                article_for_model = article
                source_language = LANGUAGES[article_language]
                if article.strip() and source_language != "eng_Latn":
                    article_for_model = translate_texts(
                        [article],
                        source_language,
                        "eng_Latn",
                        max_new_tokens=512,
                    )[0]

                tokenizer, model = load_model()
                headlines = generate_headlines(
                    article_for_model,
                    tokenizer,
                    model,
                    image_caption=image_caption,
                    number=5,
                )
                headlines = translate_headlines(headlines, LANGUAGES[output_language])

                st.subheader(f"🎯 Generated Headlines ({output_language})")

                if headlines:
                    for i, headline in enumerate(headlines, start=1):
                        safe_headline = html.escape(headline)
                        st.markdown(
                            f'<div class="headline-card"><span class="headline-index">{i:02}</span>'
                            f'<span>{safe_headline}</span></div>',
                            unsafe_allow_html=True,
                        )
                        copy_button(headline)
                else:
                    st.warning("No headline was generated. Please try another article.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

