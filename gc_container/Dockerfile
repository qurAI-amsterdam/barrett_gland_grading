FROM docker.io/nvidia/cuda:11.1.1-runtime-ubuntu20.04

## Make user algortihm and copy src and checkpoints
#RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
#
#RUN mkdir -p /opt/algorithm /input /output \
#    && chown algorithm:algorithm /opt/algorithm /input /output
#
#USER algorithm
#COPY --chown=algorithm:algorithm src/ /opt/algorithm/src/
#COPY --chown=algorithm:algorithm checkpoints/ /opt/algorithm/checkpoints/
#WORKDIR /opt/algorithm/src/
#
# Set timezone
ENV TZ=Europe/Amsterdam
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install python3.8
RUN : \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes \
    && apt-get install -y --no-install-recommends python3.8-venv \
    && apt-get install libpython3.8-dev -y \
    && apt-get clean \
    && :

# Add env to PATH
RUN python3.8 -m venv /venv
ENV PATH=/venv/bin:$PATH

# Install Git, ASAP
RUN : \
    && apt-get update \
    && apt-get -y install git \
    && apt-get -y install curl \
    && curl --remote-name --location "https://github.com/computationalpathologygroup/ASAP/releases/download/ASAP-2.0-(Nightly)/ASAP-2.0-py38-Ubuntu2004.deb" \
    && dpkg --install ASAP-2.0-py38-Ubuntu2004.deb || true \
    && apt-get -f install --fix-missing --fix-broken --assume-yes \
    && ldconfig -v \
    && apt-get clean \
    && echo "/opt/ASAP/bin" > /venv/lib/python3.8/site-packages/asap.pth \
    && rm ASAP-2.0-py38-Ubuntu2004.deb \
    && : \

# Install Python packages
RUN : \
    && pip install wheel \
#    && pip install segmentation-models-pytorch \
    && : \

# Run the pipeline
ENTRYPOINT ["tail", "-f", "/dev/null"]