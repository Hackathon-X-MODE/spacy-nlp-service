from flask import Flask, request, json

from ml.model.model import REF, MAIN_TYPE_MODEL, MOOD_MODEL, PREPARE_ORDER_SUB_TYPE_MODEL


api = Flask(__name__)

moodModel = REF[MOOD_MODEL].load()

typesModels = {}

for model in REF:
    if model != MOOD_MODEL:
        typesModels[model] = REF[model].load()


@api.route('/mood', methods=['POST'])
def get_companies():
    print(request.data.decode("utf-8"))
    return json.dumps(moodModel(request.data.decode("utf-8")).cats), 200, {
        'Content-Type': 'application/json; charset=utf-8'}


@api.route('/types', methods=['POST'])
def main_type():
    comment = request.data.decode("utf-8")
    result = []
    for currentModel in typesModels:
        result.append({
            "name": currentModel,
            "result": typesModels[currentModel](comment).cats
        })

    return json.dumps(result), 200, {'Content-Type': 'application/json; charset=utf-8'}


if __name__ == '__main__':
    api.run()
