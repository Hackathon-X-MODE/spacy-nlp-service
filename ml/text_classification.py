import spacy
import random
from spacy.util import minibatch, compounding
from spacy.training.example import Example


class TextClassification:
    def __init__(self, train_data: list, model: str):
        self.nlp = spacy.blank("ru")
        self.train_data = train_data
        self.model = model
        text_cat = self.nlp.add_pipe(self.model)#"textcat_multilabel"

        for cat in list(self.train_data[0][1]["cats"].keys()):
            text_cat.add_label(cat)
            print("Attach label for model: ", cat)

    def training(self):
        self.nlp.begin_training()
        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != self.model]
        with self.nlp.disable_pipes(*other_pipes):
            sizes = compounding(1.0, 4.0, 1.001)
            for epoch in range(9):
                random.shuffle(self.train_data)
                batches = minibatch(self.train_data, size=sizes)
                losses = {}
                for batch in batches:
                    texts, annotations = zip(*batch)
                    examples = []
                    for text, annotation in zip(texts, annotations):
                        examples.append(Example.from_dict(self.nlp.make_doc(text), annotation))
                    self.nlp.update(examples, losses=losses)
                print("Epoch:", epoch, "Loss:", losses)
        return self

    def save(self, path: str):
        self.nlp.to_disk(path)
