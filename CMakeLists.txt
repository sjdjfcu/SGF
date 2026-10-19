cmake_minimum_required(VERSION 3.13)
project(GraphSLAM)
###########
# Options #
###########
OPTION(BUILD_PYTHON "build Python bindings and Habitat MVP utilities" OFF)
OPTION(BUILD_KOBUKI_BRIDGE "build direct Kobuki JSON-lines control bridge" OFF)
OPTION(BUILD_NAVIGATION_TESTS "build native navigation-map regression tests" OFF)
IF(BUILD_NAVIGATION_TESTS)
	enable_testing()
ENDIF()
###################
# Project setting #
###################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-function" )
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
#################################
# Specify the CMake module path #
#################################
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
SET(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR} ${CMAKE_PREFIX_PATH})

#################################
# Add additional compiler flags #
#################################
IF(${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++ -Wno-deprecated-declarations -Wno-unused-function")
ENDIF()
IF(NOT MSVC_IDE)
	SET(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
	SET(CMAKE_CXX_FLAGS "-fPIC ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF()
############
# External #
############
SET(WITH_OPENCV TRUE)
SET(WITH_EIGEN TRUE)
SET(WITH_TINYPLY TRUE)
SET(WITH_JSON TRUE)
INCLUDE(UseOpenCV)
INCLUDE(Usetinyply)
INCLUDE(UseJson)
INCLUDE(UseOpenMP)
################
# Include dirs #
################
INCLUDE_DIRECTORIES(cpp/orutils/include)
######################
# Add subdirectories #
######################
ADD_SUBDIRECTORY(cpp/inseg_lib)
ADD_SUBDIRECTORY(cpp/graphslam)
IF(BUILD_PYTHON)
	ADD_SUBDIRECTORY(cpp/pybind)
ENDIF(BUILD_PYTHON)
IF(BUILD_KOBUKI_BRIDGE)
	ADD_SUBDIRECTORY(cpp/kobuki_bridge)
ENDIF(BUILD_KOBUKI_BRIDGE)
