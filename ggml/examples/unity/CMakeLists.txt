#
# unity

set(TEST_TARGET unity)
add_executable(${TEST_TARGET} unity.cpp)

target_include_directories(${TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

target_sources(${TEST_TARGET}
    PRIVATE
        model_loader.cpp
        unity_model_loader.cpp
        unity_text_decoder.cpp
)

target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)

#
# TODO: unity-quantize

# set(TEST_TARGET unity-quantize)
# add_executable(${TEST_TARGET} quantize.cpp)
# target_link_libraries(${TEST_TARGET} PRIVATE ggml common common-ggml)
