import pandas as pd

# Load dataset
df = pd.read_csv("feedback.csv")

# Keep important columns
df = df[["reviewText", "overall"]]

# Remove empty rows
df.dropna(inplace=True)

# Convert ratings into sentiment labels
def convert_sentiment(rating):

    if rating >= 4:
        return "positive"

    elif rating == 3:
        return "neutral"

    else:
        return "negative"

# Create sentiment column
df["sentiment"] = df["overall"].apply(convert_sentiment)

# Rename review column
df.rename(columns={"reviewText": "feedback"}, inplace=True)

# Keep only required columns
df = df[["feedback", "sentiment"]]

# safe sampling
sample_size = min(len(df), 5000)
df = df.sample(sample_size, random_state=42)

# Save cleaned dataset
df.to_csv("cleaned_feedback.csv", index=False)

print("Dataset prepared successfully!")
print(df.head())