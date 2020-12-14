## Build docker image
```
./build_docker.sh
```

## Use docker image
```
cd /parent-folder-containing/robocode
/location/of/this/repo/mlo-dev/launch_docker.sh
```

From the docker container
```
cd /code/robocode

# Continue to build as if running on Linux
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo -GNinja
cmake --build .
ctest
```

