FROM nvcr.io/nvidia/pytorch:21.11-py3

RUN groupadd -g 701 zdeploy && \
    useradd -g zdeploy -u 701 zdeploy -m -s /bin/bash && \
    chown -R zdeploy:zdeploy $INST_DIR /workspace

USER zdeploy

# Proxy for downloading files and uploading log to Wandb
ENV HTTP_PROXY=http://10.60.28.99:81
ENV HTTPS_PROXY=http://10.60.28.99:81
ENV no_proxy="localhost,127.0.0.0/8,10.0.0.0/8,*.infra.zalo.services,infra.zalo.services,*.zalo.services,zalo.services,zalogit2.zing.vn,nexus-repo.zapps.vn"

RUN pip install --quiet "pytorch-lightning==1.6.4" "torchmetrics" "transformers==4.17.0" "fvcore" "clean-text" "unidecode" "gdown"

RUN mkdir -p HierSpellCorrection
COPY . /workspace/HierSpellCorrection
WORKDIR /workspace/HierSpellCorrection/data

# Val dataset
RUN gdown --id 1RQ1LrhfvKuXTMJ18V_xX8WM7i0W17qHB
# Train dataset
RUN gdown --id 1nKoizh2BkHWooQGS4TdJP3M6TZKeQ9F5

RUN unzip train.zip
RUN unzip val.zip

WORKDIR /workspace/HierSpellCorrection

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]
EXPOSE 8888

# Note: DIGITS uses shared memory to share data between processes.
# For example, if you use Torch multiprocessing for multi-threaded data loaders,
# the default shared memory segment size that the container runs with may not be enough.
# Therefore, you should increase the shared memory size by issuing either:
# --ipc=host
# --shm-size=