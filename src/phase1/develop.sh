export PYTHONPATH=/home/project/app/kernel:$PYTHONPATH
gunicorn -w $(getconf _NPROCESSORS_ONLN) -b :5000 prediction:app
