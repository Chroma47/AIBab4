# Example 4.24 (Modified)
# Question Answering with Hugging Face Transformers
# Different text + multiple questions

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load a pre-trained question-answering model
model_name = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# ---------------------------------------------------------
# NEW CONTEXT: short Marie Curie biography
# ---------------------------------------------------------
context = (
    "Marie Skłodowska Curie was a Polish and naturalized-French physicist "
    "and chemist who conducted pioneering research on radioactivity. "
    "She was born on 7 November 1867 in Warsaw, Poland, and died on "
    "4 July 1934 in Passy, France. She was the first woman to win a Nobel "
    "Prize, the first person to win a Nobel Prize twice, and the only person "
    "to win a Nobel Prize in two different scientific fields. "
    "Her husband, Pierre Curie, shared the 1903 Nobel Prize in Physics with "
    "her and Henri Becquerel. In 1911, she won the Nobel Prize in Chemistry "
    "for her discovery of the elements polonium and radium."
)

# ---------------------------------------------------------
# Multiple questions to ask against the same context
# ---------------------------------------------------------
questions = [
    "What was Marie Curie's nationality?",       # multi-answer
    "When was Marie Curie born?",                # date
    "Where did Marie Curie die?",                # place
    "What elements did Marie Curie discover?",   # multi-answer
    "Who was Marie Curie's husband?",            # person
    "How many Nobel Prizes did Marie Curie win?",# numeric (inferred)
    "What is the capital of France?",            # answer NOT in context
]

# ---------------------------------------------------------
# Helper: answer a single question against the context
# ---------------------------------------------------------
def answer_question(question, context, top_k=1):
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits   = outputs.end_logits[0]

    # Simple top-1 extraction (original approach)
    start_position = torch.argmax(start_logits).item()
    end_position   = torch.argmax(end_logits).item()

    # Guard against end < start
    if end_position < start_position:
        end_position = start_position

    answer_tokens = inputs["input_ids"][0][start_position:end_position + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    # Confidence scores (softmax)
    start_conf = torch.softmax(start_logits, dim=-1)[start_position].item()
    end_conf   = torch.softmax(end_logits,   dim=-1)[end_position].item()
    confidence = start_conf * end_conf

    return answer, confidence

# ---------------------------------------------------------
# Run all questions
# ---------------------------------------------------------
print("\n" + "=" * 78)
print("CONTEXT:")
print(context)
print("=" * 78)

print(f"\n{'#':<3}{'Question':<45}{'Answer':<25}{'Conf':>8}")
print("-" * 78)

for i, q in enumerate(questions, start=1):
    ans, conf = answer_question(q, context)
    print(f"{i:<3}{q:<45}{ans:<25}{conf:>8.4f}")

print("=" * 78)