# import json
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# # import json_logging
# import numpy as np
# from loguru import logger
# from features.orchestrator import Orchestrator
# import cython_code
# import uuid
#
# app = Flask(__name__)
# cors = CORS(app, resources={r'/api/*': {'origin': '*'}})
#
# orch = Orchestrator()
# COLUMNS = ['feature{}'.format(i) for i in range(1, 42)]
#
#
# @app.route("/phase-2/prob-1/predict", methods=['POST'])
# def predict():
#     ids, drift, res = None, 0, []
#     try:
#         data = request.get_json(force=True)
#
#         # with open('./log/problem1_{}.txt'.format(uuid.uuid4()), 'w') as f:
#         #     f.write(str(data))
#
#         # logger.info("Get info")
#         ids = data.get('id')
#         columns = data.get('columns')
#
#         rows = cython_code.convert2numpyarr(data.get('rows'))
#         # rows = np.asarray(data.get('rows'))
#
#         # logger.info("Reorder features")
#         new_column_indexes = [columns.index(name) for name in COLUMNS]
#         rows = rows[:, new_column_indexes]
#
#         res = orch.predict(data=rows, model='prob1')
#         # logger.info("Return")
#
#         return jsonify(
#             {
#                 'id': ids,
#                 'predictions': res,
#                 'drift': 0
#             }
#         )
#
#     except Exception as e:
#         logger.error(e)
#         return jsonify(
#             {
#                 'id': ids,
#                 'predictions': res,
#                 'drift': 0
#             }
#         )
#
#
# @app.route("/phase-2/prob-2/predict", methods=['POST'])
# def predict_prob2():
#     ids, drift, res = None, 0, []
#     try:
#         data = request.get_json(force=True)
#
#         # with open('./log/problem2_{}.txt'.format(uuid.uuid4()), 'w') as f:
#         #     f.write(str(data))
#
#         # logger.info("Get info")
#         ids = data.get('id')
#         columns = data.get('columns')
#
#         rows = cython_code.convert2numpyarr(data.get('rows'))
#         # rows = np.asarray(data.get('rows'))
#
#         # logger.info("Reorder features")
#         new_column_indexes = [columns.index(name) for name in COLUMNS]
#         rows = rows[:, new_column_indexes]
#
#         res = orch.predict(data=rows, model='prob2')
#         # logger.info("Return")
#         return jsonify(
#             {
#                 'id': ids,
#                 'predictions': res,
#                 'drift': 0
#             }
#         )
#
#     except Exception as e:
#         logger.error(e)
#         return jsonify(
#             {
#                 'id': ids,
#                 'predictions': res,
#                 'drift': 0
#             }
#         )
#
#
# if __name__ == '__main__':
#     app.run()


from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from features.orchestrator import Orchestrator
from loguru import logger
import cython_code
import os


app = FastAPI()
orch = Orchestrator()
COLUMNS = ['feature{}'.format(i) for i in range(1, 42)]


@app.post("/phase-2/prob-1/predict")
async def predict(request: Request):
    ids, drift, res = None, 0, []
    try:
        data = await request.json()

        ids = data.get('id')
        columns = data.get('columns')

        logger.info(os.getpid())

        rows = cython_code.convert2numpyarr(data.get('rows'))

        new_column_indexes = [columns.index(name) for name in COLUMNS]
        rows = rows[:, new_column_indexes]

        res = orch.predict(data=rows, model='prob1')

        return JSONResponse(
            {
                'id': ids,
                'predictions': res,
                'drift': 0
            }
        )

    except Exception as e:
        logger.error(e)
        return JSONResponse(
            {
                'id': ids,
                'predictions': res,
                'drift': 0
            }
        )


@app.post("/phase-2/prob-2/predict")
async def predict_prob2(request: Request):
    ids, drift, res = None, 0, []
    try:
        data = await request.json()

        ids = data.get('id')
        columns = data.get('columns')

        logger.info(os.getpid())

        rows = cython_code.convert2numpyarr(data.get('rows'))

        new_column_indexes = [columns.index(name) for name in COLUMNS]
        rows = rows[:, new_column_indexes]

        res = orch.predict(data=rows, model='prob2')

        return JSONResponse(
            {
                'id': ids,
                'predictions': res,
                'drift': 0
            }
        )

    except Exception as e:
        logger.error(e)
        return JSONResponse(
            {
                'id': ids,
                'predictions': res,
                'drift': 0
            }
        )
