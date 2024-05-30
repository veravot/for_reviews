from nltk.corpus import stopwords
from pymystem3 import Mystem
import nltk
import yake
from rutermextract import TermExtractor
import matplotlib.pyplot as plt

term_extractor = TermExtractor()
nltk.download("stopwords")


def normalize_text(filename: str):
    # чтение файла из задачи
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
        # предобработка текста(
    mystem = Mystem()
    russian_stopwords = stopwords.words("russian")
    # добавляем в список стоп-слов сокращенные названия мер длины и веса, а также некоторые слова, не несущие смысла
    russian_stopwords.append('гр')
    russian_stopwords.append('г')
    russian_stopwords.append('м')
    russian_stopwords.append('мм')
    russian_stopwords.append('это')
    russian_stopwords.append('пока')
    russian_stopwords.append('очень')
    # приводим токены к нижнему регистру
    tok_text = mystem.lemmatize(text.lower())
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
    print(result)
    # записываем предобработанные отзывы в файл
    with open("Отзывы_предобработанные.txt", "w", encoding="utf-8") as f:
        f.write(result)

    # извлекаем ключевые слова
    def get_keywords(result):
        keywords = []
        keywords_count = []
        print("Эти ключевые слова получены c помощью библиотеки RuTermExtract:")
        for term in term_extractor(result):
            if term.count > 6:
                print(term.normalized, term.count)
                keywords.append(term.normalized)
                keywords_count.append(term.count)
        # Построение столбчатой диаграммы по ключевым словам c помощью библиотеки RuTermExtract
        plt.figure(figsize=(10, 6))
        plt.bar(
            range(len(keywords)), keywords_count, tick_label=keywords, color="skyblue"
        )
        plt.xlabel("Слово")
        plt.ylabel("Частота")
        plt.title("Ключевые слова")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
        # записываем ключевые слова в файл
        with open("keywords_rutermextract.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(keywords))

        # Обработка текста, поиск ключевых слов c помощью метода YAKE
        print("Эти ключевые слова получены c помощью метода YAKE:")
        language = "ru" # язык, с которым работаем
        max_ngram_size = 2 # количество слов в извлекаемом словосочетании. меняли на 1, 2, 3, 4 
        numOfKeywords = 10 # общее количество извлекаемых слов
        kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, top=numOfKeywords)
        keywords2 = kw_extractor.extract_keywords(result)
        # Выводим извлеченные ключевые слова и вероятность, отражающую степень важности ключевого слова
        for kw in keywords2:
            print(kw)
        # записываем ключевые слова в файл
        keywords2 = [str(i) for i in keywords2]
        keywords2 = " ".join(keywords2)
        with open("yake.keywords.txt", "w", encoding="utf=8") as f:
            f.write(keywords2)

    print(get_keywords(result))


print(normalize_text("Отзывы.txt"))
