# unity

add_executable(unity unity.cpp)
target_include_directories(unity PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
target_link_libraries(unity PRIVATE ggml common common-ggml kaldi-native-fbank)
target_sources(unity
    PRIVATE
        fairseq2.cpp
        model_loader.cpp
        unity_model_loader.cpp
)
