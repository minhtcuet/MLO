export PYTHONPATH=/home/project/app/kernel:$PYTHONPATH
gunicorn -w 15 -b :5000 prediction:app

