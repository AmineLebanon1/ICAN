FROM python:3.9-slim-buster
WORKDIR /app
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./requirements.txt /app
RUN pip install -r requirements.txt
RUN pip install tensorflow==2.15
RUN apt-get install -y nvidia-cuda-toolkit
RUN pip3 install tensorflow-gpu==2.12.0
COPY . .
EXPOSE 5000
# required by heroku documentation
# exposes a dynamic port to the outside world 
# port is determined by heroku
# CMD gunicorn --bind 0.0.0.0:$PORT wsgi
ENV FLASK_APP=app.py
CMD [ "python", "-m" , "flask", "run", "--host=0.0.0.0"]