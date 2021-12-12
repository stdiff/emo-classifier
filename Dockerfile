FROM python:3.9.7-slim

RUN pip install --upgrade pip

RUN useradd -ms /bin/bash --uid 1000 --gid 100 worker
USER worker
WORKDIR /tmp

ENV PATH=/home/worker/.local/bin:${PATH}
COPY --chown=worker:users dist/emo_classifier-0-py3-none-any.whl .
RUN pip install emo_classifier-0-py3-none-any.whl

EXPOSE 8000
CMD python -m emo_classifier.server