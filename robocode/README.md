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

## Push docker image to Azure Container Registry

Install azure-cli: https://docs.microsoft.com/en-us/cli/azure/

```
az login
az acr login --name your_registry_name
docker tag lisaong/robocode-dev:1.0 your_registry_name.azurecr.io/build-env/linux:1.0
docker push your_registry_name.azurecr.io/build-env/linux:1.0
```

Reference: https://docs.microsoft.com/en-us/azure/container-registry/container-registry-get-started-docker-cli
