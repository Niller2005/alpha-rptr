FROM python:3-slim-buster

RUN apt-get update && apt-get install -y \
    gcc \
    libsm6 \
    build-essential \
    cmake \
    wget \
    pkg-config \
    libx11-dev \
    libatlas-base-dev \
    libgtk-3-dev

RUN cd /tmp && \
    wget -O ta-lib.tar.gz http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar xvzf ta-lib.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && make install

ADD ./requirements.txt /tmp/requirements.txt

RUN python -m pip install -U pip

RUN pip install -r /tmp/requirements.txt

ADD ./ /opt/webapp/
WORKDIR /opt/webapp

CMD python main.py --account binanceaccount1 --exchange binance --pair BTCUSDT --strategy SMA2
