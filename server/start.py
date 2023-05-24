from flask import Flask, json, request

from ml.model.model import HelloWorld

companies = [{"id": 1, "name": "Company One"}, {"id": 2, "name": "Company Two"}]

api = Flask(__name__)

model = HelloWorld().load()


@api.route('/companies', methods=['GET'])
def get_companies():
    return json.dumps(model("Test").cats)


if __name__ == '__main__':
    api.run()
