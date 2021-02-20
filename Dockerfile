FROM jupyter/scipy-notebook:0ce64578df46

RUN conda install xgboost

RUN conda install pipenv

RUN conda install imbalanced-learn

RUN conda install seaborn

RUN conda install lime

RUN conda install hyperopt

RUN pipenv install hpsklearn

RUN conda install graphviz

RUN conda install scikit-learn

ENV PYTHONPATH "${PYTHONPATH}:/home/jovyan/work"

RUN echo "export PYTHONPATH=/home/jovyan/work" >> ~/.bashrc

WORKDIR /home/jovyan/work
