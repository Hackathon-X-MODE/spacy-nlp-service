import spacy
import random
import json
from spacy.util import minibatch, compounding
from spacy.training.example import Example

# with open('out.json', encoding='utf-8') as json_file:
#     train_data = json.load(json_file)

# Загрузка модели языка
nlp = spacy.load("ru_core_news_lg")

# Создание компонента "textcat_multilabel"
textcat = nlp.add_pipe("textcat_multilabel")

# Добавление меток классов
textcat.add_label("OTHER")
textcat.add_label("PRODUCT_DESCRIPTION")
textcat.add_label("GETTING_ORDER")
textcat.add_label("NOTIFICATION")
textcat.add_label("GOT_ORDER")
textcat.add_label("POST_BOX")
textcat.add_label("PRODUCT")
textcat.add_label("DELIVERY")

# Загрузка данных для обучения
train_data = [
    ("This is a positive sentence", {"cats": {"POSITIVE": True, "NEGATIVE": False, "NEUTRAL": False}}),
    ("This is a negative sentence", {"cats": {"POSITIVE": False, "NEGATIVE": True, "NEUTRAL": False}}),
    ("This is a neutral sentence", {"cats": {"POSITIVE": False, "NEGATIVE": False, "NEUTRAL": True}}),
    # Добавьте остальные обучающие данные здесь
]

# Компиляция модели
nlp.begin_training()
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat_multilabel']
with nlp.disable_pipes(*other_pipes):
    sizes = compounding(1.0, 4.0, 1.001)
    for epoch in range(10):
        random.shuffle(train_data)
        batches = minibatch(train_data, size=sizes)
        losses = {}
        for batch in batches:
            texts, annotations = zip(*batch)
            examples = []
            for text, annotation in zip(texts, annotations):
                examples.append(Example.from_dict(nlp.make_doc(text), annotation))
            nlp.update(examples, losses=losses)
        print("Epoch:", epoch, "Loss:", losses)

# Тестирование модели
test_text = "This is a positive sentence"
doc = nlp(test_text)
print(test_text, doc.cats)

test_text = "This is a negative sentence"
doc = nlp(test_text)
print(test_text, doc.cats)

test_text = "This is a neutral sentence"
doc = nlp(test_text)
print(test_text, doc.cats)


nlp.to_disk("test-04")
