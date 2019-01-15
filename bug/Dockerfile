FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install python3 python3-pip wget -y
RUN pip3 install gensim==3.6.0

COPY trigger.py /app/

RUN wget http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz -O data.tar.gz
RUN tar xzf data.tar.gz
RUN find 20_newsgroups -type f | xargs cat > /app/data.txt
RUN rm -rf data.tar.gz 20_newsgroups

RUN python3 --version
RUN python3 /app/trigger.py
