FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9
MAINTAINER minhtran

# make directories suited to your application
RUN mkdir -p /home/project/app
#RUN mkdir -p /home/project/app/model
WORKDIR /home/project/app

# copy and install packages for flask
COPY requirements.txt /home/project/app

#RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir pip --upgrade
RUN pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir -r requirements.txt

# copy contents from your local to your docker container
COPY . /home/project/app
COPY ./models /home/project/app/models

ENV PYTHONPATH=/home/project/app
# ENTRYPOINT ./develop.sh

EXPOSE 5000

CMD ["gunicorn", "main:app", "-w", "8", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:5000"]
