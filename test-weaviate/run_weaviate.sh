#!/bin/bash

#curl -L https://github.com/weaviate/weaviate/releases/download/v1.37.2/weaviate-v1.37.2-linux-arm64.tar.gz -o weaviate.tar.gz
#tar -xzf weaviate.tar.gz

env $(cat weaviate.env | xargs) ./weaviate --host 0.0.0.0 --port 8080 --scheme http


