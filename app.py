import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
import tensorflow as tf
import numpy as np
import json
import re
import csv
from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from lime.lime_text import LimeTextExplainer
import pandas as pd
import altair as alt

CONTEXT_SIZE = 512
MODEL_PATH = "location_model_best.keras"

EXAMPLE_TEXTS = [
    "Russia declares martial law in annexed Ukrainian territories, orders mass evacuation. This unprecedented move comes as tensions escalate in the region, with Moscow citing security concerns and the need to protect its interests.",
    "The White House announced new executive orders today as Congress debates the federal budget. The President met with senior advisers at the Oval Office to discuss domestic economic policy and infrastructure spending.",
    "Flooding in Bangladesh has displaced over a million people after record monsoon rains swept through the Ganges delta. Aid organizations are struggling to reach affected villages.",
    "Tech giants in Silicon Valley reported record quarterly earnings as AI chip demand surges. Several San Francisco-based startups announced major funding rounds driven by artificial intelligence investments.",
    "Beijing announced sweeping new regulations targeting the domestic technology sector, requiring local data storage and government oversight of algorithms used by major platforms.",
    "Wildfires continue to burn across New South Wales and Victoria, forcing thousands of residents to evacuate. Australian authorities warn the fire season could be the worst in a decade.",
    "Japanese parliament approved a new defense budget doubling military spending over the next five years, citing regional security threats from North Korea and China.",
]


from model_layers import PositionalEmbedding

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"PositionalEmbedding": PositionalEmbedding})

    tokenizer = Tokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer.enable_padding(length=CONTEXT_SIZE)
    tokenizer.enable_truncation(max_length=CONTEXT_SIZE)

    with open("unique_fips_locations.json") as f:
        unique_locations = json.load(f)

    fips_names = {}
    with open("fipsCodes.csv", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 3:
                fips_names[row[0].strip()] = row[2].strip()

    location_keys = list(unique_locations.keys())
    class_names = [fips_names.get(k, k) for k in location_keys]

    return model, tokenizer, location_keys, class_names


def make_predict_fn(model, tokenizer, mc_dropout=False, mc_samples=10):
    def predict_fn(texts):
        encodings = tokenizer.encode_batch(list(texts))
        input_ids = np.array([enc.ids for enc in encodings], dtype=np.int32)
        if mc_dropout:
            preds = np.stack([
                model(input_ids, training=True).numpy()
                for _ in range(mc_samples)
            ])
            return preds.mean(axis=0)
        return model.predict(input_ids, verbose=0, batch_size=16)

    return predict_fn


def highlight_html(text, word_weights):
    if not word_weights:
        return f'<p style="line-height:2;font-size:15px">{text}</p>'

    max_abs = max(abs(v) for v in word_weights.values()) or 1.0
    lc_weights = {k.lower(): v for k, v in word_weights.items()}
    tokens = re.split(r"(\b\w+\b)", text)
    parts = []

    for token in tokens:
        w = lc_weights.get(token.lower())
        if w is not None:
            intensity = min(abs(w) / max_abs, 1.0)
            alpha = 0.25 + intensity * 0.60
            color = f"rgba(30,180,30,{alpha:.2f})" if w > 0 else f"rgba(220,40,40,{alpha:.2f})"
            parts.append(
                f'<span title="weight: {w:+.4f}" style="background:{color};border-radius:3px;'
                f'padding:1px 4px;font-weight:600">{token}</span>'
            )
        else:
            parts.append(token)

    return '<p style="line-height:2.2;font-size:15px">' + "".join(parts) + "</p>"


st.set_page_config(page_title="Location Classifier", layout="wide")
st.title("Article Location Classifier")
st.caption(
    "Predicts the geographic location described in a news article and highlights "
    "which words are most influential in that prediction."
)

model, tokenizer, location_keys, class_names = load_resources()

with st.sidebar:
    st.header("Quick Examples")
    for ex in EXAMPLE_TEXTS:
        if st.button(ex[:55] + "...", key=ex, use_container_width=True):
            st.session_state["input_text"] = ex

    st.divider()
    st.header("Settings")
    top_n = st.slider("Top predictions to show", 3, 10, 5)
    num_features = st.slider("Words to highlight", 5, 20, 10)
    num_samples = st.slider("LIME samples", 50, 400, 150, 50,
                            help="More samples give more accurate importance scores but take longer.")

    st.divider()
    st.header("MC Dropout")
    mc_dropout = st.toggle("Enable dropout during inference", value=False)
    mc_samples = st.slider(
        "MC samples", 5, 50, 10,
        disabled=not mc_dropout,
        help="Number of stochastic forward passes to average. More passes improve accuracy but take longer.",
    )

predict_fn = make_predict_fn(model, tokenizer, mc_dropout, mc_samples)

text = st.text_area(
    "Enter article text",
    value=st.session_state.get("input_text", ""),
    height=160,
    placeholder="Paste a news article or headline here...",
)

run = st.button("Analyze", type="primary", disabled=not text.strip())

if run and text.strip():
    with st.spinner("Running prediction..."):
        probs = predict_fn([text])[0]

    top_idx = np.argsort(probs)[-top_n:][::-1]
    best_idx = top_idx[0]
    best_label = class_names[best_idx]

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        st.subheader("Predictions")

        pred_data = pd.DataFrame({
            "Location": [class_names[i] for i in top_idx],
            "Confidence": [float(probs[i]) for i in top_idx],
        })

        chart = (
            alt.Chart(pred_data)
            .mark_bar()
            .encode(
                x=alt.X("Confidence:Q", axis=alt.Axis(format=".0%"), title="Confidence"),
                y=alt.Y("Location:N", sort="-x", title=None),
                color=alt.condition(
                    alt.datum.Location == best_label,
                    alt.value("#1f77b4"),
                    alt.value("#aec7e8"),
                ),
                tooltip=[
                    alt.Tooltip("Location:N"),
                    alt.Tooltip("Confidence:Q", format=".2%"),
                ],
            )
            .properties(height=40 * top_n)
        )
        st.altair_chart(chart, use_container_width=True)

        st.dataframe(
            pred_data.style.format({"Confidence": "{:.2%}"}),
            hide_index=True,
            use_container_width=True,
        )

    with col_right:
        st.subheader(f"Word Importance for: {best_label}")

        with st.spinner(f"Running LIME with {num_samples} samples..."):
            explainer = LimeTextExplainer(class_names=class_names)
            exp = explainer.explain_instance(
                text,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=[best_idx],
            )
            feat_list = exp.as_list(label=best_idx)

        feat_df = pd.DataFrame(feat_list, columns=["Word", "Weight"]).sort_values("Weight")

        importance_chart = (
            alt.Chart(feat_df)
            .mark_bar()
            .encode(
                x=alt.X("Weight:Q", title="Impact on prediction"),
                y=alt.Y("Word:N", sort=None, title=None),
                color=alt.condition(
                    alt.datum.Weight > 0,
                    alt.value("#2ca02c"),
                    alt.value("#d62728"),
                ),
                tooltip=[
                    alt.Tooltip("Word:N"),
                    alt.Tooltip("Weight:Q", format="+.4f"),
                ],
            )
            .properties(height=max(30 * num_features, 200))
        )
        st.altair_chart(importance_chart, use_container_width=True)
        st.caption("Green = word supports this location | Red = word contradicts this location")

    st.subheader("Highlighted Text")
    st.caption(
        "Words highlighted green push the model toward the predicted location. "
        "Red words push against it. Hover a word to see its exact weight."
    )
    word_weights = dict(feat_list)
    st.markdown(highlight_html(text, word_weights), unsafe_allow_html=True)
