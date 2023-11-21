build: build/src/libggml.so

build/src/libggml.so: examples/unity/*.h examples/unity/*.cpp
	mkdir -p build
	cd build; cmake -DBUILD_SHARED_LIBS=On -DCMAKE_BUILD_TYPE=Debug ..
	cd build; make -j4 fairseq2_cpp
	find build/ -iname '*.so'

tests: build/src/libggml.so
	pytest ./*.py -s
