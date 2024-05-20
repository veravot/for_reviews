import torch
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizerFast
import pandas as pd
import matplotlib.pyplot as plt

tokenizer = BertTokenizerFast.from_pretrained("blanchefort/rubert-base-cased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained(
    "blanchefort/rubert-base-cased-sentiment", return_dict=True
)


def predict(text):
    inputs = tokenizer(
        text, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return predicted


# Чтение данных их файла
df = pd.read_excel("Отзывы.xlsx")
sample = pd.DataFrame()
sample = df.sample(224)
# Применение функции по определению тональности
sample["prediction"] = sample["Отзывы"].apply(predict)
# Запись результата в файл с отзывами
sample.to_excel("/content/Отзывы.xlsx")


# Создаем дополнительный столбец для перевода наименований
sample["Категория"] = sample["prediction"].copy()
sample["Категория"].replace([0], "нейтральный", inplace=True)
sample["Категория"].replace([1], "положительный", inplace=True)
sample["Категория"].replace([2], "отрицательный", inplace=True)
print(sample)

# Считаем количество отзывов в каждой категории
counts = sample["Категория"].value_counts()

# Создаем круговую диаграмму
plt.figure(figsize=(9, 6))
plt.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=140)
plt.title("Соотношение положительных, отрицательных и нейтральных отзывов")
plt.axis("equal")
plt.legend(title="Тип", loc="lower right")
plt.show()
