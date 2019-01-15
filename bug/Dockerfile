FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install python3 python3-pip wget -y
RUN pip3 install gensim==3.6.0

COPY download.sh /app/
COPY trigger.py /app/

RUN bash -c "cd /app; bash download.sh"
RUN python3 --version
RUN python3 /app/trigger.py
