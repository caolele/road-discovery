#!/bin/sh
docker stop rd-demo
docker rm rd-demo
docker rmi road-discovery:latest