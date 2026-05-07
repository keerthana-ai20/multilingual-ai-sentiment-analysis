import pandas as pd

from sklearn.model_selection import train_test_split

from datasets import Dataset

from transformers import (

    AutoTokenizer,
    AutoModelForSequenceClassification,

    Trainer,
    TrainingArguments
)

# Load dataset
df = pd.read_csv("cleaned_feedback.csv")

# Label encoding
label_map = {

    "negative": 0,
    "neutral": 1,
    "positive": 2
}

df["label"] = df["sentiment"].map(label_map)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(

    df["feedback"],
    df["label"],

    test_size=0.2,

    random_state=42
)

# Multilingual tokenizer
MODEL_NAME = "bert-base-multilingual-cased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Tokenize
def tokenize(batch):

    return tokenizer(

        batch["text"],

        padding=True,
        truncation=True,

        max_length=128
    )

# Datasets
train_dataset = Dataset.from_dict({

    "text": train_texts.tolist(),

    "label": train_labels.tolist()
})

test_dataset = Dataset.from_dict({

    "text": test_texts.tolist(),

    "label": test_labels.tolist()
})

train_dataset = train_dataset.map(tokenize, batched=True)

test_dataset = test_dataset.map(tokenize, batched=True)

# Load multilingual model
model = AutoModelForSequenceClassification.from_pretrained(

    MODEL_NAME,

    num_labels=3,

    ignore_mismatched_sizes=True
)
# Training settings
training_args = TrainingArguments(

    output_dir="./multi_results",

    eval_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,

    per_device_eval_batch_size=4,

    num_train_epochs=1,

    weight_decay=0.01,

    logging_dir="./multi_logs",

    save_strategy="epoch"
)

# Trainer
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset
)

# Train
trainer.train()

# Save model
model.save_pretrained("./multilingual_model")

tokenizer.save_pretrained("./multilingual_model")

print("Multilingual model trained successfully!")