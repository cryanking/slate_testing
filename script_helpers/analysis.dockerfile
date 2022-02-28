FROM jupyter/scipy-notebook:notebook-6.4.8

RUN mamba install --quiet --yes gcc && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

RUN pip install --quiet --no-cache-dir xgboost && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# tag cryanking/slate_analysis:0.1
