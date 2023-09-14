#
# unity

set(TEST_TARGET unity)
add_executable(${TEST_TARGET} unity.cpp)
target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# TODO: unity-quantize

# set(TEST_TARGET unity-quantize)
# add_executable(${TEST_TARGET} quantize.cpp)
# target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
