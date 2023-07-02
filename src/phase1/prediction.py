import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import json_logging
from loguru import logger
from features.orchestrator import Orchestrator, cal_psi

app = Flask(__name__)
cors = CORS(app, resources={r'/api/*': {'origin': '*'}})

json_logging.init_flask(enable_json=True)

orch = Orchestrator()


@app.route("/phase-1/prob-1/predict", methods=['POST'])
def predict():
    ids, drift, res = None, 0, []
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            data = json.loads(data)

        ids = data.get('id')
        rows = data.get('rows')
        columns = data.get('columns')

        res = list(orch.predict(data=rows, columns=columns, model='prob1'))
        drift = cal_psi(res)

        return jsonify(
            {
                'id': ids,
                'predictions': res,
                'drift': 1 if drift > 0.25 else 0
            }
        )

    except Exception as e:
        logger.error(e)
        return jsonify(
            {
                'id': ids,
                'predictions': res,
                'drift': drift
            }
        )


@app.route("/phase-1/prob-2/predict", methods=['POST'])
def predict_prob2():
    ids, drift, res = None, 0, []
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            data = json.loads(data)

        ids = data.get('id')
        rows = data.get('rows')
        columns = data.get('columns')

        res = list(orch.predict(data=rows, columns=columns, model='prob2'))
        drift = cal_psi(res)

        return jsonify(
            {
                'id': ids,
                'predictions': res,
                'drift': 1 if drift > 0.25 else 0
            }
        )

    except Exception as e:
        logger.error(e)
        return jsonify(
            {
                'id': ids,
                'predictions': res,
                'drift': drift
            }
        )

if __name__ == '__main__':
    app.run()