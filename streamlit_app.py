# streamlit_app.py

import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import io
# Optional: For PDF reading
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict

# --- Dashboard Layout ---
st.set_page_config(
    page_title="Emotion Detection Dashboard",
    layout="wide",
    page_icon=":smiley:"
)

# --- Theme handling (persistent) ---
if 'theme' not in st.session_state:
    st.session_state['theme'] = 'light'

# Sidebar theme toggle (persisted)
with st.sidebar:
    st.markdown("## Appearance")
    theme_choice = st.radio("Theme", ["light", "dark"], index=0 if st.session_state['theme']=='light' else 1)
    if theme_choice != st.session_state['theme']:
        st.session_state['theme'] = theme_choice
        st.rerun()  # apply immediately

# Apply theme CSS globally
if st.session_state['theme'] == "dark":
    st.markdown(
        """
        <style>
        /* dark theme */
        .reportview-container, .main, .block-container {background-color: #0e1116; color: #e6eef6;}
        .stTextArea textarea, .stTextInput input {background-color: #111316; color: #e6eef6;}
        .sentence-card {background: #121416 !important; color: #e6eef6; border-radius:12px; padding:12px; box-shadow: 0 6px 18px rgba(0,0,0,0.4);}
        .emotion-label {color: #7fb3ff !important;}
        </style>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        /* light theme (default) */
        .reportview-container, .main, .block-container {background-color: #f8f9fa; color: #0b1220;}
        .stTextArea textarea, .stTextInput input {background-color: #ffffff; color: #0b1220;}
        .sentence-card {background: #fff !important; color: #0b1220; border-radius:12px; padding:12px; box-shadow: 0 4px 10px rgba(0,0,0,0.06);}
        .emotion-label {color: #4F8BF9 !important;}
        </style>
        """, unsafe_allow_html=True
    )

# --- Cached model loader for performance ---
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("./sentiment_model")
    tokenizer = AutoTokenizer.from_pretrained("./sentiment_model")
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

sentiment_pipeline = load_model()

# Label names
label_names = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

# Caching helper: token count
@st.cache_data
def get_token_count(text: str) -> int:
    if not text:
        return 0
    try:
        enc = sentiment_pipeline.tokenizer(text, add_special_tokens=False)
        return len(enc.get("input_ids", []))
    except Exception:
        return len(text.split())

# Caching helper: extract text once per uploaded file (by bytes)
@st.cache_data
def extract_text_from_file_bytes(file_bytes: bytes, mime_type: str) -> str:
    if not file_bytes:
        return ""
    try:
        if mime_type == "text/plain":
            return file_bytes.decode("utf-8", errors="ignore")
        if mime_type == "application/pdf" and PyPDF2 is not None:
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception:
        return ""
    return ""

# small helper to bridge UploadedFile -> cached byte extractor
def extract_text_from_file(uploaded_file) -> str:
    """Read uploaded file once and use cached extractor."""
    if uploaded_file is None:
        return ""
    try:
        file_bytes = uploaded_file.read()
        return extract_text_from_file_bytes(file_bytes, uploaded_file.type)
    except Exception:
        return ""

# Session history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# emoji / colors
emoji_map = {
    "sadness": "😢", "joy": "😄", "love": "😍",
    "anger": "😠", "fear": "😨", "surprise": "😲"
}
color_map = {
    "Sadness": "#3498db",
    "Joy": "#f1c40f",
    "Love": "#e84393",
    "Anger": "#e74c3c",
    "Fear": "#8e44ad",
    "Surprise": "#16a085"
}

# --- Sidebar Navigation & actions ---
st.sidebar.image("https://img.icons8.com/color/96/000000/happy--v2.png", width=80)
st.sidebar.title("Emotion Detection")
st.sidebar.markdown("Fine-tuned DistilBERT on the Emotion dataset.")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["Dashboard", "Session History"])
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown("- Enter text or upload a `.txt`/`.pdf` file.")
st.sidebar.markdown("- Each line/sentence will be analyzed separately.")
st.sidebar.markdown("- Use the theme toggle and Insert Example button for convenience.")

with st.sidebar.expander("Model Info & Settings"):
    st.markdown(f"**Model path:** `./sentiment_model`")
    st.markdown(f"**Max tokens per input:** 512")
    st.markdown(f"**Classes:** {', '.join(label_names)}")
    st.markdown("**Dataset:** [Emotion](https://huggingface.co/datasets/emotion)")
    st.markdown("**Version:** 1.0")

def get_example() -> str:
    # If you want to load from a file:
    try:
        with open("sample_input1.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # fallback example text
        return """I can't believe I got the job! This is the best day of my life.
Even though I tried my best, I still failed. I feel so disappointed and sad.
My friends threw me a surprise party, and I was completely shocked!
I love spending time with my family, they make me feel so warm inside.
The thunderstorm last night was terrifying. I couldn't sleep at all.
Why did you break my favorite mug? I'm really angry about this.
I feel nothing. Everything seems dull and empty.
She smiled at me, and my heart skipped a beat.
I studied all night, but the exam was still so hard. I'm frustrated.
When I heard the news, I was both happy and scared at the same time.
"""

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
# new: trigger flag used to auto-start analysis when example/file is provided
if 'trigger_analyze' not in st.session_state:
    st.session_state['trigger_analyze'] = False

with st.sidebar:
    if st.button("Insert Example Input"):
        st.session_state['user_input'] = get_example()
        st.session_state['trigger_analyze'] = True
        st.rerun()

# --- Main Dashboard Page ---
if page == "Dashboard":
    st.title("Emotion Detection Dashboard")
    st.caption("Analyze emotions in your text or documents. Visualize per-sentence predictions and summary statistics.")

    # keep file uploader and example controls in the left column
    col_left, col_right = st.columns([3, 1])
    with col_left:
        uploaded_file = st.file_uploader("Or upload a .txt or .pdf file", type=["txt", "pdf"])
        # auto-trigger analysis when a file is uploaded
        if uploaded_file is not None:
            st.session_state['trigger_analyze'] = True

        if st.button("Insert Example Input"):
            st.session_state['user_input'] = get_example()
            st.session_state['trigger_analyze'] = True
            st.rerun()

    # Chat-style input area (bottom-docked). Uses st.chat_input when available.
    st.markdown(
        """
        <style>
        /* bottom-docked chat input */
        .chat-input-wrapper {
            position: fixed;
            left: 8%;
            right: 8%;
            bottom: 18px;
            z-index: 9999;
            background: transparent;
        }
        .chat-input-box .stTextArea textarea { border-radius: 12px !important; padding: 12px !important; min-height: 64px !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    analyze = False
    user_message = None

    # prefer native chat_input if available
    if hasattr(st, "chat_input"):
        user_message = st.chat_input("Type or paste text and press Enter...")
        if user_message:
            st.session_state['user_input'] = user_message
            analyze = True
    else:
        # fallback: bottom text_area + Analyze button
        with st.container():
            st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)
            c1, c2 = st.columns([12,1])
            with c1:
                user_text = st.text_area("Type text to analyze...", value=st.session_state.get('user_input',''), key="chat_fallback", label_visibility="collapsed", height=80)
            with c2:
                if st.button("Analyze", use_container_width=True):
                    st.session_state['user_input'] = user_text
                    analyze = True
            st.markdown('</div>', unsafe_allow_html=True)
    # check trigger flag (example or file upload)
    if st.session_state.get("trigger_analyze", False):
        analyze = True
        st.session_state["trigger_analyze"] = False

    # If there's a file uploaded, prefer file contents; otherwise use last chat input
    if analyze:
        text_data = st.session_state.get('user_input', "").strip()
        if uploaded_file is not None:
            file_text = extract_text_from_file(uploaded_file)
            if file_text:
                text_data = file_text.strip()
            else:
                st.warning("Could not extract text from the uploaded file.")

        if text_data:
            # split into non-empty lines for per-sentence analysis
            sentences = [line.strip() for line in text_data.split('\n') if line.strip()]
            preds_list = sentiment_pipeline(sentences)  # uses your cached [`load_model`](streamlit_app.py)

            summary_emotions, summary_confidences, summary_sentences = [], [], []
            emotion_counts = {emo: 0 for emo in label_names}

            for idx, (sentence, preds) in enumerate(zip(sentences, preds_list)):
                best = max(preds, key=lambda x: x['score'])
                emotion = label_names[int(best['label'].split('_')[-1])]
                confidence = best['score']
                token_count = get_token_count(sentence)
                truncated = token_count > 512

                summary_emotions.append(emotion.capitalize())
                summary_confidences.append(confidence)
                summary_sentences.append(f"Sentence {idx+1}")
                emotion_counts[emotion] += 1

            # KPI row and tabs (unchanged UI logic)
            col1, col2, col3 = st.columns(3)
            col1.metric("Sentences", len(sentences))
            dominant = max(emotion_counts, key=emotion_counts.get)
            col2.metric("Dominant Emotion", dominant.capitalize(), emoji_map.get(dominant, ""))
            avg_conf = sum(summary_confidences) / len(summary_confidences)
            col3.metric("Avg Confidence", f"{avg_conf:.2%}")

            tab1, tab2, tab3 = st.tabs(["Sentence Analysis", "Summary", "Export"])

            with tab1:
                st.subheader("Sentence Analysis")
                for idx, (sentence, emo, conf) in enumerate(zip(sentences, summary_emotions, summary_confidences)):
                    token_count = get_token_count(sentence)
                    with st.container():
                        c1, c2, c3 = st.columns([3, 1, 1])
                        c1.markdown(f"**Sentence {idx+1}:** {sentence}")
                        c2.metric("Prediction", f"{emo} {emoji_map.get(emo.lower(),'')}", f"{conf:.2%}")
                        c3.markdown(f"**Tokens:** {token_count}")
                        st.progress(conf)

            with tab2:
                st.subheader("Summary Statistics")
                # horizontal bar chart and pie chart (unchanged)
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=summary_sentences,
                    x=summary_confidences,
                    orientation='h',
                    text=[f"{emo} ({conf:.2%})" for emo, conf in zip(summary_emotions, summary_confidences)],
                    hovertext=sentences,
                    marker_color=[color_map.get(emo, "#888") for emo in summary_emotions]
                ))
                fig.update_layout(xaxis=dict(title="Confidence", range=[0, 1]), yaxis=dict(title="Input Sentences"), height=400)
                st.plotly_chart(fig, use_container_width=True)

                pie_fig = go.Figure(data=[go.Pie(labels=list(emotion_counts.keys()), values=list(emotion_counts.values()), hole=0.4)])
                pie_fig.update_traces(marker=dict(colors=[color_map.get(e.capitalize(), "#888") for e in emotion_counts.keys()]))
                st.plotly_chart(pie_fig, use_container_width=True)

            with tab3:
                st.subheader("Export Results")
                df = pd.DataFrame({"Sentence": sentences, "Predicted Emotion": summary_emotions, "Confidence": summary_confidences})
                st.download_button("Download Results as CSV", df.to_csv(index=False), file_name="emotion_results.csv")

            # save session history
            st.session_state['history'].append({"sentences": sentences, "emotions": summary_emotions, "confidences": summary_confidences})
        else:
            st.warning("Please enter some text or upload a valid file.")
# Session History page
elif page == "Session History":
    st.title("Session History")
    history = st.session_state.get('history', [])
    if not history:
        st.info("No previous runs yet.")
    else:
        for idx, entry in enumerate(reversed(history[-5:])):
            with st.expander(f"Run {len(history)-idx} - {len(entry['sentences'])} sentences"):
                for s, e, c in zip(entry["sentences"], entry["emotions"], entry["confidences"]):
                    st.markdown(f"- *{s[:80]}...*: **{e}** ({c:.2%})")
            st.markdown("---")