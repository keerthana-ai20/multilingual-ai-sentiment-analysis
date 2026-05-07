from flask import Flask, render_template, request, jsonify

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import torch
import torch.nn.functional as F

from collections import Counter

from wordcloud import WordCloud

import matplotlib.pyplot as plt

import os

app = Flask(__name__)

# =========================
# MODEL CONFIG
# =========================

MODEL_PATH = "./multilingual_model"

# Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

# =========================
# LABELS
# =========================

labels = {

    0: "negative",

    1: "neutral",

    2: "positive"
}

# =========================
# HISTORY STORAGE
# =========================

feedback_history = []

sentiment_history = []

# =========================
# HOME ROUTE
# =========================

@app.route("/")
def home():

    return render_template("index.html")

# =========================
# PREDICTION ROUTE
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    try:

        feedback = request.form["feedback"]

        # Empty input check

        if feedback.strip() == "":

            return jsonify({

                "error": "Empty feedback"
            })

        # =========================
        # TOKENIZATION
        # =========================

        inputs = tokenizer(

            feedback,

            return_tensors="pt",

            truncation=True,

            padding=True,

            max_length=128
        )

        # =========================
        # MODEL PREDICTION
        # =========================

        with torch.no_grad():

            outputs = model(**inputs)

        probabilities = F.softmax(

            outputs.logits,

            dim=1
        )

        prediction = torch.argmax(
            probabilities
        ).item()

        sentiment = labels[prediction]

        confidence = round(

            probabilities[0][prediction].item() * 100,

            2
        )

        # =========================
        # STORE HISTORY
        # =========================

        feedback_history.append(feedback)

        sentiment_history.append(sentiment)

        # =========================
        # SENTIMENT COUNTS
        # =========================

        counts = Counter(sentiment_history)

        # =========================
        # WORD CLOUD GENERATION
        # =========================

        all_text = " ".join(feedback_history)

        if all_text.strip() != "":

            # Tamil/Hindi/Unicode support

            font_path = "fonts/NotoSansTamil-VariableFont_wdth,wght.ttf"

            wordcloud = WordCloud(

                width=1000,

                height=500,

                background_color="white",

                collocations=False,

                font_path=font_path
            ).generate(all_text)

            # Save image

            wordcloud_path = os.path.join(

                "static",

                "wordcloud.png"
            )

            plt.figure(figsize=(12,6))

            plt.imshow(

                wordcloud,

                interpolation="bilinear"
            )

            plt.axis("off")

            plt.tight_layout()

            plt.savefig(

                wordcloud_path,

                bbox_inches="tight"
            )

            plt.close()

        # =========================
        # RETURN JSON
        # =========================

        return jsonify({

            "sentiment": sentiment,

            "confidence": confidence,

            "positive": counts.get("positive", 0),

            "negative": counts.get("negative", 0),

            "neutral": counts.get("neutral", 0),

            "wordcloud": "/static/wordcloud.png"
        })

    except Exception as e:

        print("ERROR:", e)

        return jsonify({

            "error": str(e)
        })

# =========================
# RUN FLASK
# =========================

if __name__ == "__main__":

    app.run(debug=True)