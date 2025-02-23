# Dockerfile

# Ubuntu 22.04와 CUDA 11.8 런타임 사용
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# 로케일 설정
RUN apt-get update && \
    apt-get install -y locales

RUN localedef -i en_US -f UTF-8 en_US.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV LANGUAGE en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LC_ALL en_US.UTF-8
ENV LC_CTYPE en_US.UTF-8
ENV LC_MESSAGES en_US.UTF-8

# 필수 패키지 설치
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-dev \
        libmariadb-dev \
        libpq-dev \
        net-tools \
        wget \
        nano \
        vim \
        lsof \
        iputils-ping \
        python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Anaconda 설치
WORKDIR /tmp
COPY ./Anaconda3-2024.10-1-Linux-x86_64.sh /tmp
RUN bash Anaconda3-2024.10-1-Linux-x86_64.sh -b -p /root/anaconda3 && \
    rm Anaconda3-2024.10-1-Linux-x86_64.sh

# PATH 환경 변수 설정
ENV PATH="/root/anaconda3/bin:${PATH}"

# Conda 초기화 및 Jupyter 설치
RUN /root/anaconda3/bin/conda init bash && \
    /root/anaconda3/bin/conda install -y jupyterlab jupyter

# Jupyter 설정 파일 생성
RUN jupyter notebook --generate-config

# Jupyter 설정 수정
## hash 값 빈칸으로 변경
RUN echo "c.ServerApp.password = u'sha1:f4f03e647cee:97bba2b638c552146197fbaac4566bc91bfcab8d'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.ServerApp.notebook_dir = '/data'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.ServerApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.ServerApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.ServerApp.terminado_settings = { 'shell_command': ['bash'] }" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.ServerApp.max_buffer_size = 1000000000000" >> /root/.jupyter/jupyter_notebook_config.py

# Python 심볼릭 링크 설정
RUN rm /usr/bin/python3 && ln -s /root/anaconda3/bin/python /usr/bin/python3 && \
    ln -s /usr/bin/python3 /usr/local/bin/python

# 작업 디렉토리 설정
WORKDIR /data

ENTRYPOINT ["/root/anaconda3/bin/jupyter", "lab"]
