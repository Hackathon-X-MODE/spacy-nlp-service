import json
from abc import ABC, abstractmethod

import spacy
from spacy import Language

from ml.text_classification import TextClassification


class BasicModel:
    def __init__(self, model_dir: str, train_data_path: str, basic_model='textcat_multilabel'):
        self.model_dir = model_dir
        self.train_data_path = train_data_path
        self.basic_model = basic_model

    def traning(self):
        with open(self.train_data_path, encoding='utf-8') as json_file:
            train_data = json.load(json_file)
        model = TextClassification(train_data, self.basic_model)
        model.training().save(self.model_dir)

    def load(self):
        return spacy.load(self.model_dir)


MOOD_MODEL = 'MOOD'
MAIN_TYPE_MODEL = 'MAIN_TYPE'
PREPARE_ORDER_SUB_TYPE_MODEL = 'PREPARE_ORDER_SUB_TYPE'

REF = {
    MOOD_MODEL: BasicModel(
        'models/mood', 'train_data/mood.json', 'textcat'
    ),
    MAIN_TYPE_MODEL: BasicModel(
        'models/main-type-model', 'train_data/main-type.json', 'textcat_multilabel'
    ),
    PREPARE_ORDER_SUB_TYPE_MODEL: BasicModel(
        'models/prepare-order-sub-type-model', 'train_data/prepare-order-sub-type.json', 'textcat_multilabel'
    ),
}
