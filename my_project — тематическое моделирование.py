from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
from nltk.corpus import stopwords
from pymystem3 import Mystem
import nltk

nltk.download("stopwords")


# извлекаем ключевые слова
def keywords(filename: str):
    # чтение файла и запись построчно в список
    reviews = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            text = f.readline()
            reviews.append(text.rstrip())
    reviews2 = []
    # предобработка отзывов
    for review in reviews:
        mystem = Mystem()
        russian_stopwords = stopwords.words("russian")
        # добавляем в список стоп-слов сокращенные названия мер длины и веса, а также некоторые слова, не несущие смысла
        russian_stopwords.extend(['гр', 'г', 'м', 'мм', 'это', 'пока','очень'])
        # приводим токены к нижнему регистру
        tok_text = mystem.lemmatize(review.lower())
        # исключаем пробелы, стоп-слова и знаки пунктуации
        clean_tokens = []
        for tokens in tok_text:
            if (
                tokens != " "
                and tokens not in russian_stopwords
                and tokens.strip().isalpha() == True
            ):
                clean_tokens.append(tokens)
        # объединяем токены назад в текст
        result = " ".join(clean_tokens)
        reviews2.append(result)
    print(reviews2)
    # тематическое моделирование
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reviews2)
    n_components = 3     # выбор количества тем, меняли их количество, подробнее в отчете 
    lsa = TruncatedSVD(n_components)
    lsa_matrix = lsa.fit_transform(tfidf_matrix)
    terms = np.array(vectorizer.get_feature_names_out())
    for i, topic in enumerate(lsa.components_):
        top_terms_idx = topic.argsort()[-6:][::-1]  # Получаем топ-6 слов в каждой теме, количество слов тоже брадли разное
        top_terms = terms[top_terms_idx]
        print(f"Topic {i + 1}: {', '.join(top_terms)}")


print(keywords("/content/Отзывы.txt"))
