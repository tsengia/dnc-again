FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye
ADD requirements.txt requirements.txt
ADD lightning_requirements.txt lightning_requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install -r lightning_requirements.txt