SET(targetname libGraphSLAM)

INCLUDE(UseEigen3)
INCLUDE(UseOpenCV)

FILE(GLOB sources "*.cpp")
FILE(GLOB headers "*.h")

ADD_LIBRARY(${targetname} ${sources} ${headers})
TARGET_INCLUDE_DIRECTORIES(${targetname}
        PUBLIC ..)
target_compile_definitions(${targetname}
        PUBLIC PSLAM_SOURCE_DIR=\"${PROJECT_SOURCE_DIR}\")
INCLUDE(Linktinyply)
INCLUDE(LinkJson)

IF(BUILD_NAVIGATION_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../tests/navigation_floor_support_test.cpp")
        add_executable(navigation_floor_support_test ../tests/navigation_floor_support_test.cpp)
        set_target_properties(navigation_floor_support_test PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
        target_link_libraries(navigation_floor_support_test PRIVATE libGraphSLAM inseg_lib)
        add_test(NAME navigation_floor_support COMMAND navigation_floor_support_test)
    endif()
    add_executable(navigation_graph_occupancy_test ../tests/navigation_graph_occupancy_test.cpp)
    set_target_properties(navigation_graph_occupancy_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_link_libraries(navigation_graph_occupancy_test PRIVATE libGraphSLAM inseg_lib)
    add_test(NAME navigation_graph_occupancy COMMAND navigation_graph_occupancy_test)
ENDIF()
