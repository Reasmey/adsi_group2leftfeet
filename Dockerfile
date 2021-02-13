FROM jupyter/scipy-notebook:0ce64578df46

RUN conda install xgboost

RUN conda install pipenv

RUN conda install imbalanced-learn

RUN conda install seaborn

RUN conda install lime

RUN conda install hyperopt

RUN conda install graphviz

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc

WORKDIR /home/jovyan/work
