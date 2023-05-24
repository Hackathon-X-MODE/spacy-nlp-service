FROM continuumio/miniconda3
RUN mkdir -p /usr/spacy
COPY . /usr/spacy/

RUN conda create -n venv
SHELL ["conda", "run", "-n", "venv", "/bin/bash", "-c"]
RUN conda install -y -c conda-forge spacy
RUN conda install -y -c conda-forge spacy-transformers
RUN pip install flask
###################

WORKDIR /usr/spacy
EXPOSE 5000

ENTRYPOINT ["conda", "run", "-n", "venv", "flask","run","--host=0.0.0.0"]