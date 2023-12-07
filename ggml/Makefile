build: build/examples/unity/libfairseq2_cpp.so ggml/build/bin/unity

build/examples/unity/libfairseq2_cpp.so: Makefile examples/unity/*.h examples/unity/*.cpp src/ggml*.c
	mkdir -p build
	cd build; cmake\
		-DGGML_OPENBLAS=ON \
	  -DBUILD_SHARED_LIBS=On \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CXX_FLAGS="-g2 -fno-omit-frame-pointer" \
	  -DTRACY_ENABLE=ON \
	  ..
	cd build; make -j4 fairseq2_cpp
	find build/ -iname '*.so'


ggml/build/bin/unity: Makefile examples/unity/*.h examples/unity/*.cpp src/ggml*.c
	mkdir -p build
	cd build; cmake\
		-DGGML_OPENBLAS=ON \
	  -DBUILD_SHARED_LIBS=On \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CXX_FLAGS="-g2 -fno-omit-frame-pointer" \
	  -DTRACY_ENABLE=ON \
	  ..
	cd build; make -j4 unity
	find build/ -iname '*.so'


tests: build/src/libggml.so
	pytest ./*.py -s

build/src/libggml_cuda.so: Makefile examples/unity/*.h examples/unity/*.cpp
	mkdir -p build
	cd build; cmake\
	  -DGGML_CUBLAS=ON \
	  -DBUILD_SHARED_LIBS=On \
	  -DCMAKE_BUILD_TYPE=Release \
	  -DCMAKE_CXX_FLAGS="-g2" \
	  ..
	cd build; make -j4 ggml
	mv build/src/libggml.so build/src/libggml_cuda.so
	find build/ -iname '*.so'

cuda_tests: build/src/libggml_cuda.so
	sed -i 's/lib_base_name = "ggml"/lib_base_name = "ggml_cuda"/' third_party_ggml.py
	pytest ./*.py -s
	sed -i 's/lib_base_name = "ggml_cuda"/lib_base_name = "ggml"/' third_party_ggml.py
