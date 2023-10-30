build: build/src/libggml.so build/bin/unity

build/src/libggml.so: examples/unity/*.h examples/unity/*.cpp
	mkdir -p build
	cd build; cmake -DBUILD_SHARED_LIBS=On -DCMAKE_BUILD_TYPE=Debug ..
	cd build; make -j4 ggml
	find build/ -iname '*.so'

build/bin/unity: examples/unity/*.h examples/unity/*.cpp
	mkdir -p build
	cd build; cmake ..
	cd build; make -j4 unity

run: build/bin/unity
	$< --model examples/unity/models/unity-large/ggml-model.bin

tests: build/src/libggml.so
	pytest ./*.py -s
