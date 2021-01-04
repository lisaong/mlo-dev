call conda activate robocode
conan remote add robocode http://robocode-packages.eastus.cloudapp.azure.com:8081/artifactory/api/conan/RoboCode-Packages
conan user -p RoboCode2020 -r robocode admin

REM conda install -c conda-forge openblas
call "d:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
call "d:\win\Program Files (x86)\IntelSWTools\compilers_and_libraries\windows\mkl\bin\mklvars.bat" intel64

set REPO_ROOT=d:\win\repos\rb\
pip install -r %REPO_ROOT%requirements.txt
pip install -r %REPO_ROOT%external\onnxbenchmarks\requirements.txt
pip install -r %REPO_ROOT%tools\notebooks\requirements.txt
jupyter nbextension enable --py --sys-prefix qgrid
jupyter nbextension enable --py --sys-prefix widgetsnbextension
jupyter nbextension install --sys-prefix --py vega3
jupyter nbextension enable --py --sys-prefix vega3

REM call cmake -DUSE_BLAS=1 -DUSE_MKL=1 -DCMAKE_BUILD_TYPE=RelWithDebInfo -DLLVM_LIT_ARGS=-vv -DROBOCODE_DISABLE_LOWERING_SNAPSHOTS=YES "-GVisual Studio 16 2019" -Ax64 ..
REM call cmake --build . --config RelWithDebInfo --target check-all -- /m:4
REM call cmake --build . --config RelWithDebInfo -- /m:4
REM call cmake --build . --target install -- /m:4

REM https://tinyurl.com/y3dm3h86
pip install numpy==1.19.3

REM python 3.8
pip install -i https://test.pypi.org/simple/ ort-nightly
pip install torch transformers tokenizers