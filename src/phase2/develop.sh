export PYTHONPATH=/home/project/app/kernel:$PYTHONPATH
#gunicorn -w 7 -b :5000 prediction:app

uvicorn prediction:app --host 0.0.0.0 --port 5000 --workers 14