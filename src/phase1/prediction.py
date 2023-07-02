import json_logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from loguru import logger

from features.orchestrator import Orchestrator, cal_psi_pro1, cal_psi_pro2

app = FastAPI()
json_logging.init_fastapi(enable_json=True)

orch = Orchestrator()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    id: str
    predictions: list[float]
    drift: int


@app.post("/phase-1/prob-1/predict", response_model=PredictionResponse)
async def predict_prob1(request: Request):
    try:
        data = await request.json()
        ids = data.get('id')
        rows = data.get('rows')
        columns = data.get('columns')

        predictions = orch.predict(data=rows, columns=columns, model='prob1')
        drift = cal_psi_pro1(predictions)

        response = PredictionResponse(
            id=ids,
            predictions=predictions,
            drift=1 if drift > 0.25 else 0
        )

        return response
    except Exception as e:
        logger.error(e)
        return PredictionResponse()


@app.post("/phase-1/prob-2/predict", response_model=PredictionResponse)
async def predict_prob2(request: Request):
    try:
        data = await request.json()
        ids = data.get('id')
        rows = data.get('rows')
        columns = data.get('columns')

        predictions = orch.predict(data=rows, columns=columns, model='prob2')
        drift = cal_psi_pro2(predictions)

        response = PredictionResponse(
            id=ids,
            predictions=predictions,
            drift=1 if drift > 0.25 else 0
        )

        return response
    except Exception as e:
        logger.error(e)
        return PredictionResponse()


if __name__ == '__main__':
    uvicorn.run(app)
