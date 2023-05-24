from abc import ABC, abstractmethod

import spacy
from spacy import Language

from ml.text_classification import TextClassification


class Model(ABC):
    @abstractmethod
    def traning(self):
        pass

    @abstractmethod
    def load(self):
        pass


class HelloWorld(Model):

    def __init__(self):
        self.model_dir = 'models/hello-world'

    def traning(self):
        train_data = [
            ("This is a positive sentence", {"cats": {"POSITIVE": True, "NEGATIVE": False, "NEUTRAL": False}}),
            ("This is a negative sentence", {"cats": {"POSITIVE": False, "NEGATIVE": True, "NEUTRAL": False}}),
            ("This is a neutral sentence", {"cats": {"POSITIVE": False, "NEGATIVE": False, "NEUTRAL": True}}),
            # Добавьте остальные обучающие данные здесь
        ]
        model = TextClassification(train_data)
        model.training().save('server/' + self.model_dir)

    def load(self):
        return spacy.load(self.model_dir)
