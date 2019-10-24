#!/bin/sh
BASE=$(dirname $0)
cd $BASE/

TAG="train_tfrecords:dev"
docker build -t=$TAG ./

