import os
import random

import pandas as pd
import streamlit as st
import sklearn

from predictor import load_bundle, predict

MODEL_PATH = "fake_news_model.joblib"
MIN_CONFIDENCE = 0.70   # below this the app says "Uncertain"
MIN_WORDS = 25          # fewer cleaned words than this = not enough text

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown(
    """
    <style>
    .result-card {border-radius: 14px; padding: 22px 26px; margin: 8px 0 18px 0;}
    .result-real {background: #e8f7ef; border: 1px solid #1f9d63;}
    .result-fake {background: #fdecec; border: 1px solid #d64545;}
    .result-uncertain {background: #fff6e0; border: 1px solid #e0a100;}
    .result-uncertain h2 {color: #8a6100;}
    .result-card h2 {margin: 0 0 4px 0;}
    .result-real h2 {color: #167a4c;}
    .result-fake h2 {color: #b83232;}
    .result-card p {margin: 0; color: #333;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


# ---------------------------------------------------------------- model check
if not os.path.exists(MODEL_PATH):
    st.title("📰 Fake News Detector")
    st.error(
        f"`{MODEL_PATH}` not found. Run the training notebook first, download the "
        "model file and keep it in the same folder as `app.py`."
    )
    st.stop()

bundle = get_bundle()
metrics = bundle.get("metrics", {})
samples = bundle.get("samples", {"Real": [], "Fake": []})

# ---------------------------------------------------------------------- state
for key in ("headline", "article"):
    st.session_state.setdefault(key, "")


def load_sample(kind):
    pool = samples.get(kind, [])
    if pool:
        s = random.choice(pool)
        st.session_state["headline"] = s["title"]
        st.session_state["article"] = s["text"]
        st.session_state["sample_kind"] = kind


def clear_inputs():
    st.session_state["headline"] = ""
    st.session_state["article"] = ""
    st.session_state.pop("sample_kind", None)


# -------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "AI-Powered Fake News Detection using Multimodal NLP. "
        "This demo is the **text-only baseline** (TF-IDF + Logistic Regression). "
        "The image module is the next stage."
    )
    if metrics:
        st.subheader("Model performance")
        st.caption("Held-out test set")
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        c2.metric("F1-score", f"{metrics['f1']*100:.2f}%")
        c1.metric("Precision", f"{metrics['precision']*100:.2f}%")
        c2.metric("Recall", f"{metrics['recall']*100:.2f}%")
    st.subheader("Try a sample")
    st.button("Random real article", on_click=load_sample, args=("Real",))
    st.button("Random fake article", on_click=load_sample, args=("Fake",))
    st.button("Clear", on_click=clear_inputs)
    st.divider()
    st.caption(
        "⚠️ Decision-support tool only. It cannot confirm whether a story is "
        "factually true. Always verify with trusted sources."
    )
    trained_ver = bundle.get("sklearn_version")
    if trained_ver and trained_ver != sklearn.__version__:
        st.warning(
            f"Model trained with scikit-learn {trained_ver}, but this machine has "
            f"{sklearn.__version__}. If you see errors, install the same version."
        )

# ----------------------------------------------------------------------- main
st.title("📰 Fake News Detector")
st.write("Paste a headline and (optionally) the article text, then press **Analyze**.")
st.info(
    "**How to read the result:** this tool judges the *writing style* of a news item, "
    "based on patterns learned from ISOT news articles (2016-17, mostly US politics). "
    "It **cannot verify facts** or recent events, so a convincing-sounding false claim "
    "can still be marked Real. Treat it as a screening signal, not proof."
)

left, right = st.columns([3, 2], gap="large")

with left:
    st.text_input("Headline", key="headline", placeholder="Enter the news headline")
    st.text_area(
        "Article text (optional, but gives better results)",
        key="article",
        height=240,
        placeholder="Paste the article body here...",
    )
    analyze = st.button("🔍 Analyze", type="primary")

with right:
    image = st.file_uploader("Associated image (optional)", type=["png", "jpg", "jpeg"])
    if image is not None:
        st.image(image, caption="Uploaded image")
        st.info(
            "Image received. The image-analysis (CNN) module is still under "
            "development, so this prediction is based on **text only**."
        )

if analyze:
    result = predict(bundle, st.session_state["headline"], st.session_state["article"])
    if result is None:
        st.warning("Please enter a headline or some article text first.")
    else:
        too_short = result["n_words_input"] < MIN_WORDS
        low_conf = result["confidence"] < MIN_CONFIDENCE
        uncertain = too_short or low_conf

        if uncertain:
            css, icon, title = "result-uncertain", "⚠️", "UNCERTAIN"
            note = f"Model leans <b>{result['label']}</b> ({result['confidence']*100:.1f}%), but not strongly enough to call it."
        elif result["label"] == "Real":
            css, icon, title = "result-real", "✅", "Likely REAL (in writing style)"
            note = f"Model confidence: <b>{result['confidence']*100:.1f}%</b>"
        else:
            css, icon, title = "result-fake", "🚩", "Likely FAKE (in writing style)"
            note = f"Model confidence: <b>{result['confidence']*100:.1f}%</b>"

        st.markdown(
            f"""<div class="result-card {css}">
                <h2>{icon} {title}</h2>
                <p>{note}</p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.progress(result["p_real"], text=f"Real {result['p_real']*100:.1f}%  |  Fake {result['p_fake']*100:.1f}%")

        if too_short:
            st.warning(
                "Not enough text for a reliable result (a headline alone is usually too short). "
                "Paste the article body for a better prediction."
            )
        elif low_conf:
            st.warning("The model's confidence is low, so please verify this news with trusted sources.")

        kind = st.session_state.get("sample_kind")
        if kind and st.session_state["headline"]:
            st.caption(f"This was loaded as a sample from the test set (true label: {kind}).")

        st.subheader("Why this prediction?")
        st.caption("Words in your text that pushed the model the most (larger = stronger influence).")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pushing toward REAL**")
            if result["toward_real"]:
                df = pd.DataFrame(result["toward_real"], columns=["word", "influence"]).set_index("word")
                st.bar_chart(df, color="#1f9d63")
            else:
                st.write("No strong words found.")
        with c2:
            st.markdown("**Pushing toward FAKE**")
            if result["toward_fake"]:
                df = pd.DataFrame(result["toward_fake"], columns=["word", "influence"]).set_index("word")
                st.bar_chart(df, color="#d64545")
            else:
                st.write("No strong words found.")
