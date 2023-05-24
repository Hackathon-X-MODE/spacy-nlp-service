import spacy

print("start")
# Загрузка сохраненной модели
model_dir = "test-04"
nlp = spacy.load(model_dir)
print("done")
# Использование модели для классификации текста
test_data = [
    "На Windows 10 не работает",
    "Слишком дорагая доставка",
    "Товар разбит",
    "У конкурентов сервис лучше!",
    "Постамат лагает!",
    "Почините постамат",
    "такое впечатление что специально положили не тот товар  больше в пункты выдачи заказывать не буду только потерянное время да еще нахамили в поддержке ужас"
]


def my_filtering_function(pair):
    key, value = pair
    if value >= 0.49:
        return True  # keep pair in the filtered dictionary
    else:
        return False  # filter pair out of the dictionary


for val in test_data:
    doc = nlp(val)
    print(val, dict(filter(my_filtering_function, doc.cats.items())))
