# freezing to the correct cuda installation
FROM brunneis/python:3.8.3-ubuntu-20.04

# Adding application code to the container
ADD . /root/ajnaboiz

# Setting the non-interactive frontend for apt-get
ARG DEBIAN_FRONTEND=noninteractive

# Installing necessary dependencies
RUN apt-get clean && apt-get update && \
    apt-get install -y curl git libgl1-mesa-glx libglib2.0-0

# Installing x11-apps
RUN apt install -y x11-apps

# Installing software-properties-common for add-apt-repository
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa

# Updating and installing python3-pip
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    build-essential \
    python3-pip \
    libqt5gui5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# # CONDA
# RUN curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
#     chmod +x ~/miniconda.sh && \
#     ~/miniconda.sh -b -p /opt/conda && \
#     rm ~/miniconda.sh && \
#     /opt/conda/bin/conda install -y python=3.9 conda-build pyyaml numpy ipython cython typing typing_extensions mkl mkl-include ninja && \
#     /opt/conda/bin/conda clean -ya



# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


    

# # Configuring default Python version to be 3.8
# RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
#     update-alternatives --install /usr/bin/python python /usr/bin/python3.8 2 && \
#     update-alternatives --set python /usr/bin/python3.8 && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
#     update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
#     update-alternatives --set python3 /usr/bin/python3.8

# # Install pip for Python 3.8
# RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
#     python3.8 get-pip.py && \
#     rm get-pip.py

# Installing PyTorch with CUDA support
# RUN python3.8 -m pip install torch==1.13.1+cu117 torchvision -f https://download.pytorch.org/whl/torch_stable.html

# Copying requirements file and installing Python packages
COPY environment.yml environment.yml
# RUN python3.8 -m pip install -r requirements.txt

RUN /opt/conda/bin/conda env create -f environment.yml

