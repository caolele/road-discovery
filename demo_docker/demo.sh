
#!/bin/sh
# Hosting OS
printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     MACHINE=Linux;;
    Darwin*)    MACHINE=Mac;;
    CYGWIN*)    MACHINE=Cygwin;;
    MINGW*)     MACHINE=MinGw;;
    *)          MACHINE="UNKNOWN:${unameOut}"
esac
echo "Runtime OS: "${MACHINE}

if [ -x "$(command -v docker)" ]; then
    echo "Docker is Installed."
else
    echo "Docker is not installed. Install Docker first!"
    exit 1
fi

#BASEDIR=$(dirname "$0")

echo "Build docker image ..."
docker build -t road-discovery .
if [ $? != 0 ]; then
    printf "\033[1;31;40mError\033[0m: Failed to build docker image: road-discovery.\n"
    exit 1
else
    printf "\033[0;34;40m[DONE]\033[0m\n"
fi

echo "Entering demo ..."
docker run -it --name rd-demo -v $(pwd)/mount:/dvol road-discovery:latest