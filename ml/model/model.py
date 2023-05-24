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
GETTING_ORDER_SUB_TYPE_MODEL = 'GETTING_ORDER_SUB_TYPE'
GOT_ORDER_SUB_TYPE_MODEL = 'GOT_ORDER_SUB_TYPE'
PRODUCT_SUB_TYPE_MODEL = 'PRODUCT_SUB_TYPE'
POST_BOX_SUB_TYPE_MODEL = 'POST_BOX_SUB_TYPE'
DELIVERY_SUB_TYPE_MODEL = 'DELIVERY_SUB_TYPE'
NOTIFICATION_SUB_TYPE_MODEL = 'NOTIFICATION_SUB_TYPE'

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
    GETTING_ORDER_SUB_TYPE_MODEL: BasicModel(
        'models/getting-order-sub-type-model', 'train_data/getting-order-sub-type.json', 'textcat_multilabel'
    ),
    GOT_ORDER_SUB_TYPE_MODEL: BasicModel(
        'models/got-order-sub-type-model', 'train_data/got-order-sub-type.json', 'textcat_multilabel'
    ),
    PRODUCT_SUB_TYPE_MODEL: BasicModel(
        'models/product-sub-type-model', 'train_data/product-sub-type.json', 'textcat_multilabel'
    ),
    POST_BOX_SUB_TYPE_MODEL: BasicModel(
        'models/postbox-sub-type-model', 'train_data/postbox-sub-type.json', 'textcat_multilabel'
    ),
    DELIVERY_SUB_TYPE_MODEL: BasicModel(
        'models/delivery-sub-type-model', 'train_data/delivery-sub-type.json', 'textcat_multilabel'
    ),
    NOTIFICATION_SUB_TYPE_MODEL: BasicModel(
        'models/notification-sub-type-model', 'train_data/notification-sub-type.json', 'textcat_multilabel'
    ),
}
