import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import json_logging
from loguru import logger
from features.orchestrator import Orchestrator, cal_psi_pro1, cal_psi_pro2
from cachetools import TTLCache

import warnings

cache = TTLCache(maxsize=100, ttl=60)

warnings.filterwarnings("ignore")

app = FastAPI()
json_logging.init_fastapi(enable_json=True)

orch = Orchestrator()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/phase-1/prob-1/predict")
async def predict(request: Request):
    ids, drift, res = None, 0, []
    try:
        data = await request.json()
        if not isinstance(data, dict):
            data = json.loads(data)

        ids = data.get('id')

        # Check if the response is cached
        cached_response = cache.get(ids)
        if cached_response:
            return cached_response

        rows = data.get('rows')
        columns = data.get('columns')

        res = list(orch.predict(data=rows, columns=columns, model='prob1'))
        drift = cal_psi_pro1(res)

        response = {
            'id': ids,
            'predictions': res,
            'drift': 1 if drift > 0.25 else 0
        }

        # Cache the response
        cache[ids] = response

        return response

    except Exception as e:
        logger.error(e)
        return {
            'id': ids,
            'predictions': res,
            'drift': drift
        }


@app.post("/phase-1/prob-2/predict")
async def predict_prob2(request: Request):
    ids, drift, res = None, 0, []
    try:
        data = await request.json()
        if not isinstance(data, dict):
            data = json.loads(data)

        ids = data.get('id')

        # Check if the response is cached
        cached_response = cache.get(ids)
        if cached_response:
            return cached_response

        rows = data.get('rows')
        columns = data.get('columns')

        res = list(orch.predict(data=rows, columns=columns, model='prob2'))
        drift = cal_psi_pro2(res)

        response = {
            'id': ids,
            'predictions': res,
            'drift': 1 if drift > 0.25 else 0
        }

        # Cache the response
        cache[ids] = response

        return response

    except Exception as e:
        logger.error(e)
        return {
            'id': ids,
            'predictions': res,
            'drift': drift
        }


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app)
