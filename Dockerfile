FROM python:3.9.7-slim

RUN apt-get update
RUN apt-get install -y curl
RUN pip install --upgrade pip

RUN useradd -ms /bin/bash --uid 1000 --gid 100 worker
USER worker
CMD whoami
WORKDIR /tmp

COPY dist/emo_classifier-0-py3-none-any.whl .
RUN pip install --user emo_classifier-0-py3-none-any.whl
RUN mkdir script
COPY script/server.py script/
CMD python script/server.py
