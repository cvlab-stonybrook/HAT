FROM bitnami/python:3.10

WORKDIR /app

ENV HTTP_PORT=4000

RUN apt-get update \
    && apt-get -y install gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/*

COPY ./common ./common
COPY ./configs ./configs
COPY ./hat ./hat
COPY ./model_server.py ./model_server.py
COPY ./requirements.txt ./requirements.txt

RUN python -m pip install --no-cache -U pip \
    && python -m pip install --no-cache -r requirements.txt

RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
RUN chmod +x ./hat/pixel_decoder/ops/make.sh \
    && cd ./hat/pixel_decoder/ops \
    && ./make.sh

EXPOSE $HTTP_PORT

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:4000", "--pythonpath", ".", "--access-logfile", "-", "model_server:app"]