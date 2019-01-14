wget http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz -O data.tar.gz
tar xzf data.tar.gz
find 20_newsgroups -type f | xargs cat > data.txt
