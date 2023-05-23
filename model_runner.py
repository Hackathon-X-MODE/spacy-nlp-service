from ml.text_classification import TextClassification

train_data = [
    ("This is a positive sentence", {"cats": {"POSITIVE": True, "NEGATIVE": False, "NEUTRAL": False}}),
    ("This is a negative sentence", {"cats": {"POSITIVE": False, "NEGATIVE": True, "NEUTRAL": False}}),
    ("This is a neutral sentence", {"cats": {"POSITIVE": False, "NEGATIVE": False, "NEUTRAL": True}}),
    # Добавьте остальные обучающие данные здесь
]

model = TextClassification(train_data)

model.training().save('models/hello-world')