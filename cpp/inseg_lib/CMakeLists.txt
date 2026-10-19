INCLUDE(UseEigen3)
INCLUDE(UseOpenCV)

set(targetname inseg_lib)
add_library(${targetname} STATIC
    camera.cpp
    disjoint_set.cpp
    image_pyramid.cpp
    inseg_core.cpp
    lib.cpp
    map.cpp
    reconstruction_runtime.cpp
    segmentation.cpp
    surfel.cpp)

target_include_directories(${targetname}
    PUBLIC ${PROJECT_SOURCE_DIR}/cpp)
target_compile_options(${targetname} PRIVATE -O3)
target_compile_definitions(${targetname} PRIVATE NDEBUG)
INCLUDE(LinkEigen3)
INCLUDE(LinkOpenCV)
