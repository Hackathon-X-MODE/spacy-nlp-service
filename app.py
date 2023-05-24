from flask import Flask, request, json

from ml.model.model import REF, MAIN_TYPE_MODEL, MOOD_MODEL, PREPARE_ORDER_SUB_TYPE_MODEL

companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

api = Flask(__name__)

model = REF[MOOD_MODEL].load()

typesModels = {
    MAIN_TYPE_MODEL: REF[MAIN_TYPE_MODEL].load(),
    PREPARE_ORDER_SUB_TYPE_MODEL: REF[PREPARE_ORDER_SUB_TYPE_MODEL].load(),
}


@api.route('/mood', methods=['POST'])
def get_companies():
    print(request.data.decode("utf-8"))
    return json.dumps(model(request.data.decode("utf-8")).cats), 200, {
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
