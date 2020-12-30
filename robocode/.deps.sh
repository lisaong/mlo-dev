conan remote add robocode http://robocode-packages.eastus.cloudapp.azure.com:8081/artifactory/api/conan/RoboCode-Packages
conan user -p RoboCode2020 -r robocode admin

pip install -r /code/requirements.txt
pip install -r /code/external/onnxbenchmarks/requirements.txt
pip install -r /code/tools/notebooks.requirements.txt
jupyter nbextension enable --py --sys-prefix qgrid
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension install --sys-prefix --py vega3
jupyter nbextension enable --py --sys-prefix vega3

sudo apt-get update
sudo apt-get install -y gdb
