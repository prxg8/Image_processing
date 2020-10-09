#!/bin/bash

#doing specific results folder for various machines
name=$HOSTNAME
if [[ "$name" == "Atlas" ]]; then
	resultsFolder="/home/aaronjflood/Stuff/PINE/Simulations/AtlasResults"
else
	resultsFolder="$PWD"
fi

docker run -it \
   --user=$(id -u) \
   -e DISPLAY=$DISPLAY \
   -e QT_X11_NO_MITSHM=1\
   -p 5000:5000 \
   --workdir=/app \
   --volume="$PWD":/app \
   --volume="/etc/group:/etc/group:ro" \
   --volume="/etc/passwd:/etc/passwd:ro" \
   --volume="/etc/shadow:/etc/shadow:ro" \
   --volume="/etc/sudoers.d:/etc/sudoers.d:ro" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   geomancer-base python $1 $2 $3 $4 $5 $6 $7 $8 $9
