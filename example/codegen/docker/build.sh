#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILDROOT=$DIR/..
 
cd $BUILDROOT
 
CONTAINER="lowinli98/fastgpt-codegen"   #替换成你的容器名称
VERSION=`git describe --abbrev=0 --tags`
 
IMAGE_NAME="${CONTAINER}:${TAG}"
 
cmd="docker build -t $IMAGE_NAME -f $DIR/dockerfile $BUILDROOT"
echo $cmd
eval $cmd

docker push $IMAGE_NAME