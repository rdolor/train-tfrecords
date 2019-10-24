FROM python:3.6-slim-stretch as pytrain-base
ADD requirements.txt requirements.txt
RUN apt-get update -qq && apt-get install -yqq --no-install-recommends vim &&  apt-get install -yqq gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    #Clean-up
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean
ADD . .
#WORKDIR "/"
