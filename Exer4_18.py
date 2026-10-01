# Example 4.22 (Modified)
# NLTK Text Analysis - Modernized version
# Different text: excerpt from the "Machine Learning" Wikipedia article

import nltk

# Download the required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
nltk.download("averaged_perceptron_tagger_eng")
nltk.download("maxent_ne_chunker")
nltk.download("maxent_ne_chunker_tab")
nltk.download("words")

# ---------------------------------------------------------
# NEW TEXT: Machine Learning (Wikipedia excerpt)
# ---------------------------------------------------------
sentence = """
Machine learning (ML) is a field of study in artificial intelligence
concerned with the development and study of statistical algorithms that
can learn from data and generalize to unseen data, and thus perform tasks
without explicit instructions.

Within a subdiscipline of artificial intelligence known as machine learning,
statistical methods are used to train computer systems to solve problems.

Since the 1950s, researchers such as Alan Turing and Arthur Samuel have
explored whether computers could be programmed to learn from experience.
Modern machine learning has been applied successfully in fields such as
medicine, finance, and natural language processing, and it powers many
widely used services from Google, Microsoft, and OpenAI.
"""

# 1. Tokenization
tokens = nltk.word_tokenize(sentence)

print("\n--- TOKENS ---")
print(tokens)

# 2. Part-of-Speech tagging
tagged = nltk.pos_tag(tokens)

print("\n--- POS TAGGING ---")
print(tagged)

# 3. Named Entity Recognition
entities = nltk.chunk.ne_chunk(tagged)

print("\n--- NAMED ENTITIES ---")
print(entities)

# ---------------------------------------------------------
# 4. EXTRA: Print only the named entities in a readable list
# ---------------------------------------------------------
print("\n--- NAMED ENTITIES (flattened) ---")
for chunk in entities:
    if hasattr(chunk, "label"):
        text = " ".join(c[0] for c in chunk)
        print(f"{chunk.label():<10} {text}")

# ---------------------------------------------------------
# 5. EXTRA: Quick POS frequency summary
# ---------------------------------------------------------
from collections import Counter
pos_counts = Counter(tag for _, tag in tagged)
print("\n--- POS FREQUENCY (top 15) ---")
for tag, count in pos_counts.most_common(15):
    print(f"{tag:<6} {count}")