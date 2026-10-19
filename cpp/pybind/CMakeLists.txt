find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

INCLUDE(UseOpenCV)
INCLUDE(UseEigen3)

set(targetname sgf_habitat_core)
pybind11_add_module(${targetname} MODULE
    sgf_module.cpp
    hot_path_bindings.cpp
)
target_link_libraries(${targetname} PRIVATE libGraphSLAM inseg_lib)
target_include_directories(${targetname} PRIVATE
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/cpp/orutils/include
)
INCLUDE(LinkEigen3)
INCLUDE(LinkOpenCV)
set_target_properties(${targetname} PROPERTIES
    OUTPUT_NAME "_sgf"
    LIBRARY_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/python/sgf_habitat"
)
