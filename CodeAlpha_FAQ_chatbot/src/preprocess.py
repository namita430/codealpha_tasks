import nltk
import string
# download tokenizer automatically (no warning)
nltk.download("punkt", quiet=True)
def clean_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in string.punctuation]
    return " ".join(tokens)