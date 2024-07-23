#! /bin/bash 

XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

docker run -it \
  --env DISPLAY=$DISPLAY \
  --privileged \
  --net=host \
  -v /dev:/dev \
  --volume $XAUTH:/root/.Xauthority \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  --volume="/home/forzapbldesktop/Downloads/ajnadata:/root/ajnaboiz/data" \
  --volume="/home/forzapbldesktop/idnet:/root/ajnaboiz/idnet" \
  --name="idnet_cnt" \
  --detach \
  --gpus all \
  --rm \
  --shm-size=1gb \
  idnet \
