"""Shared text cleaning + prediction logic for the Fake News Detection project.

Used by BOTH the training notebook and the Streamlit app, so the text is
cleaned in exactly the same way during training and during prediction.
"""
import re
import string

import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

STOP_WORDS = set(ENGLISH_STOP_WORDS)

# ISOT real news starts with a source giveaway such as "WASHINGTON (Reuters) -".
# If we leave it in, the model just learns "has Reuters tag => Real", which is
# not real fake-news detection. We strip it out.
_DATELINE = re.compile(r"^[^\-\u2013\u2014]{0,80}\(reuters\)\s*[-\u2013\u2014]?\s*", re.I)
_PUNCT = re.compile("[%s]" % re.escape(string.punctuation))


def clean_text(text):
    text = str(text) if text is not None else ""
    text = _DATELINE.sub("", text, count=1)
    text = text.lower()
    text = re.sub(r"\breuters\b", " ", text)
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>+", " ", text)
    text = _PUNCT.sub(" ", text)
    text = re.sub(r"\w*\d\w*", " ", text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)


def build_content(title, body=""):
    """Headline + article body -> one cleaned string (same as training)."""
    return (clean_text(title) + " " + clean_text(body)).strip()


def load_bundle(path="fake_news_model.joblib"):
    return joblib.load(path)


def predict(bundle, title, body="", top_k=8):
    """Return a dict with label, confidence and the words that drove the decision.

    Class 1 = Real, class 0 = Fake (same as in training).
    Returns None if there is no usable text.
    """
    cleaned = build_content(title, body)
    if not cleaned:
        return None
    vec, model = bundle["vectorizer"], bundle["model"]
    X = vec.transform([cleaned])
    classes = list(model.classes_)
    p_real = float(model.predict_proba(X)[0][classes.index(1)])
    p_fake = 1.0 - p_real

    names = vec.get_feature_names_out()
    coef = model.coef_[0]
    row = X.tocoo()
    contrib = sorted(
        ((names[j], float(v * coef[j])) for j, v in zip(row.col, row.data)),
        key=lambda x: x[1],
    )
    toward_fake = [(w, -s) for w, s in contrib if s < 0][:top_k]
    toward_real = [(w, s) for w, s in reversed(contrib) if s > 0][:top_k]

    return {
        "label": "Real" if p_real >= 0.5 else "Fake",
        "confidence": max(p_real, p_fake),
        "p_real": p_real,
        "p_fake": p_fake,
        "toward_real": toward_real,
        "toward_fake": toward_fake,
        "n_terms": int(row.nnz),   # how many known words were found in the text
        "n_words_input": len(cleaned.split()),
    }
