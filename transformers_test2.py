# Example 4.24
# Question Answering with Hugging Face Transformers

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load a pre-trained question-answering model
model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Question and context
question = "What is the name of the company?"

context = (
    "We created Biox Systems Ltd company back in the year of 2000."
)

# Convert the text into tokens
inputs = tokenizer(
    question,
    context,
    return_tensors="pt"
)

# Run the model
with torch.no_grad():
    outputs = model(**inputs)

# Find the answer's start and end positions
start_position = torch.argmax(outputs.start_logits)
end_position = torch.argmax(outputs.end_logits)

# Convert the tokens back into text
answer_tokens = inputs["input_ids"][0][
    start_position:end_position + 1
]

answer = tokenizer.decode(
    answer_tokens,
    skip_special_tokens=True
)

print("Question:", question)
print("Context:", context)
print("Answer:", answer)