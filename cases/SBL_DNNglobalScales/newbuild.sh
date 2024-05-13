cd ../../; rm -r build; mkdir build; cd build; 
cmake -DUSEMPI=TRUE -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make -j |& tee compile.log

