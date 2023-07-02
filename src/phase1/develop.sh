export PYTHONPATH=/home/project/app/kernel:$PYTHONPATH
#gunicorn -w 8 -b :5000 prediction:app
gunicorn -w 8 -b :5040 prediction:app

