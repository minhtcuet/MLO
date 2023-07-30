export PYTHONPATH=/home/project/app/kernel:$PYTHONPATH
gunicorn -w 7 -b :5000 prediction:app

