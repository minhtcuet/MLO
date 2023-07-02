export PYTHONPATH=/home/project/app/kernel:$PYTHONPATH
gunicorn -w 8 -b :5000 prediction:app

