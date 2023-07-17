import json
from flask import Flask, request, jsonify
from flask_cors import CORS
# import json_logging
import numpy as np
from loguru import logger
from features.orchestrator import Orchestrator
import cython_code

app = Flask(__name__)
cors = CORS(app, resources={r'/api/*': {'origin': '*'}})

orch = Orchestrator()


@app.route("/phase-2/prob-1/predict", methods=['POST'])
def predict():
    ids, drift, res = None, 0, []
    try:
        data = request.get_json(force=True)

        logger.info("Get info")
        ids = data.get('id')
        rows = cython_code.convert2numpyarr(data.get('rows'))
        res = orch.predict(data=rows, model='prob1')
        logger.info("Return")

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

        logger.info("Get info")
        ids = data.get('id')
        rows = cython_code.convert2numpyarr(data.get('rows'))

        res = orch.predict(data=rows, model='prob2')
        logger.info("Return")
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
