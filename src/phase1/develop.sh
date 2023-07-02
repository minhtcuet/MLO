export PYTHONPATH=/home/project/app/kernel:$PYTHONPATH

#gunicorn --pythonpath /home/project/app/run -w 4 -b :5000 run:app
 gunicorn -w $(getconf _NPROCESSORS_ONLN) -b :5000 prediction:app