# unity
add_library(fairseq2_cpp)
target_include_directories(fairseq2_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
target_link_libraries(fairseq2_cpp PRIVATE ggml kaldi-native-fbank)
target_sources(fairseq2_cpp
    PRIVATE
        fairseq2.cpp
        model_loader.cpp
)
add_executable(unity unity.cpp)
find_package(PkgConfig REQUIRED)
pkg_check_modules(SNDFILE REQUIRED IMPORTED_TARGET sndfile)
target_link_libraries(unity PRIVATE ggml PkgConfig::SNDFILE)
target_sources(unity
    PRIVATE
        fairseq2.cpp
        model_loader.cpp
)
