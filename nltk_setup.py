import nltk
nltk.data.path.append("nltk_data")
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("movie_reviews")
print(nltk.data.find("tokenizers/punkt"))