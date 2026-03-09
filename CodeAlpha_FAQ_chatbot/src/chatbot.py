import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import clean_text
class FAQChatbot:
    def __init__(self, faq_file):
        with open(faq_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.questions = [clean_text(item["question"]) for item in data]
        self.answers = [item["answer"] for item in data]
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)
    def get_response(self, user_input):
        user_input = clean_text(user_input)
        user_vec = self.vectorizer.transform([user_input])
        similarity = cosine_similarity(user_vec, self.question_vectors)
        best_score = similarity.max()
        best_index = similarity.argmax()
        if best_score < 0.3:
            return "Sorry, I do not understand the question."
        return self.answers[best_index]