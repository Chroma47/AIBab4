# Example 4.23 (Modified)
# Sentiment Analysis using Hugging Face Transformers
# Testing multiple positive and negative sentences

from transformers import pipeline

# Create sentiment-analysis pipeline
# Default model: distilbert-base-uncased-finetuned-sst-2-english
classifier = pipeline("sentiment-analysis")

# ---------------------------------------------------------
# Sentences to analyze: 6 positive, 6 negative
# ---------------------------------------------------------
sentences = [
    # ----- POSITIVE -----
    "This is a good movie.",
    "I absolutely loved the food at that restaurant.",
    "The new update made the app much faster and more reliable.",
    "Her performance in the play was brilliant and deeply moving.",
    "The hotel staff were friendly and the room was spotless.",
    "What a wonderful surprise — thank you so much!",

    # ----- NEGATIVE -----
    "This is a terrible movie.",
    "The food was cold and the service was rude.",
    "The new update broke everything and I lost all my data.",
    "His performance in the play was dull and forgettable.",
    "The hotel staff were unfriendly and the room was filthy.",
    "What a horrible surprise — I am extremely disappointed.",
]

# ---------------------------------------------------------
# Run sentiment analysis on each sentence
# ---------------------------------------------------------
print("\n" + "=" * 78)
print(f"{'#':<3}{'Label':<10}{'Score':>8}   Sentence")
print("=" * 78)

results = classifier(sentences)

for i, (sent, res) in enumerate(zip(sentences, results), start=1):
    label = res["label"]
    score = res["score"]
    print(f"{i:<3}{label:<10}{score:>8.4f}   {sent}")

# ---------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------
pos = [r for r in results if r["label"] == "POSITIVE"]
neg = [r for r in results if r["label"] == "NEGATIVE"]

print("\n" + "=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"Total sentences        : {len(results)}")
print(f"Predicted POSITIVE     : {len(pos)}")
print(f"Predicted NEGATIVE     : {len(neg)}")
if pos:
    print(f"Avg POSITIVE score     : {sum(r['score'] for r in pos)/len(pos):.4f}")
if neg:
    print(f"Avg NEGATIVE score     : {sum(r['score'] for r in neg)/len(neg):.4f}")