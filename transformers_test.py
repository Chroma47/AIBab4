# Example 4.23
# Sentiment Analysis using Hugging Face Transformers

from transformers import pipeline

# Create sentiment-analysis pipeline
classifier = pipeline("sentiment-analysis")

# Analyze the text
result = classifier("This is a good movie.")

# Display the result
print(result)