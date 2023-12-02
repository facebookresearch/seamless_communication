# unity.cpp

## Introduction
[GGML](https://github.com/ggerganov/ggml) is an open source library in C to enable large model inference on various hardware platforms. We implemented unity.cpp in ggml. Now it supports SeamlessM4T model for X2T tasks - Speech-to-text translation (S2TT), Acoustic speech recognition (ASR), Text-to-text translation (T2TT).  

The project is still active in development. Contributions are welcome!

## Build
To build the interactive console for S2TT & ASR, 
```

cd seamless_communication/ggml
mkdir build; cd build
cmake \
    -DGGML_OPENBLAS=ON \
    -DBUILD_SHARED_LIBS=On \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CXX_FLAGS="-g2 -fno-omit-frame-pointer" \
    ..
make -j4 unity # Interactive Console

```
Note that `-DGGML_OPENBLAS=ON` is not necessary on macOS.

For more build commands see [Makefile](Makefile). 

## CLI usage
Command to launch an interactive console for S2TT & ASR, note that the model already includes vocabulary needed to detokenize. 
```
OPENBLAS_NUM_THREADS=8 ./bin/unity --model seamlessM4T_medium.ggml
```
In the console, enter the path of local waveform file and target language, separated by space. Note that the first run would include some “warm up” time so could be slow. 

Converted ggml models could be downloaded from 
|SeamlessM4T_large | SeamlessM4T_medium | 
|-------- | -------- | 
| [model](https://dl.fbaipublicfiles.com/seamless/models/seamlessM4T_large.ggml) | [model](https://dl.fbaipublicfiles.com/seamless/models/seamlessM4T_medium.ggml) |  

## Fairseq2 model conversion 
Models from fairseq2 checkpoints could be converted to ggml automatically with [ggml_convert.py](ggml_convert.py). 
```
python ggml_convert.py -m MODEL_NAME
```
where MODEL_NAME corresponds to asset cards in fairseq2 / seamless_communication, e.g. seamlessM4T_medium, seamlessM4T_large

## Python bindings
We also utilize ggml python bindings for better dev experience. For examples of running unity.cpp in python, refer to tests in [test_unity_cpp.py](test_unity_cpp.py). 

## [Optional]Dependencies
### OpenBLAS
We strongly suggest building with OpenBLAS, as we've seen 8x speedup on test machine. 

### libsndfile
This is needed only for the console to load waveform, but not the library.

