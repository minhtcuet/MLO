import json
from flask import Flask, request, jsonify
from flask_cors import CORS
# import json_logging
from loguru import logger
from features.orchestrator import Orchestrator

app = Flask(__name__)
cors = CORS(app, resources={r'/api/*': {'origin': '*'}})

# json_logging.init_flask(enable_json=True)

orch = Orchestrator()


@app.route("/phase-2/prob-1/predict", methods=['POST'])
def predict():
    ids, drift, res = None, 0, []
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            data = json.loads(data)

        ids = data.get('id')
        rows = data.get('rows')
        columns = data.get('columns')

        res = orch.predict(data=rows, columns=columns, model='prob1')

        return jsonify(
            {
                'id': ids,
                'predictions': res,
                'drift': 0
            }
        )

    except Exception as e:
        logger.error(e)
        return jsonify(
            {
                'id': ids,
                'predictions': res,
                'drift': 0
            }
        )


@app.route("/phase-2/prob-2/predict", methods=['POST'])
def predict_prob2():
    ids, drift, res = None, 0, []
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            data = json.loads(data)

        ids = data.get('id')
        rows = data.get('rows')
        columns = data.get('columns')

        res = orch.predict(data=rows, columns=columns, model='prob2')

        return jsonify(
            {
                'id': ids,
                'predictions': res,
                'drift': 0
            }
        )

    except Exception as e:
        logger.error(e)
        return jsonify(
            {
                'id': ids,
                'predictions': res,
                'drift': 0
            }
        )


if __name__ == '__main__':
    app.run()
