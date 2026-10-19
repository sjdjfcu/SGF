add_executable(sgf-kobuki-json-bridge kobuki_json_bridge.cpp)

set(KOBUKI_INSTALL_PREFIX "" CACHE PATH "Standalone kobuki_core install prefix")
if(KOBUKI_INSTALL_PREFIX)
    target_include_directories(sgf-kobuki-json-bridge PRIVATE
        "${KOBUKI_INSTALL_PREFIX}/include"
        "${KOBUKI_INSTALL_PREFIX}/include/eigen3"
    )
    target_link_libraries(sgf-kobuki-json-bridge PRIVATE
        "${KOBUKI_INSTALL_PREFIX}/lib/libkobuki_core.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_devices.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_mobile_robot.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_geometry.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_linear_algebra.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_formatters.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_threads.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_type_traits.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_time.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_exceptions.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_time_lite.so"
        "${KOBUKI_INSTALL_PREFIX}/lib/libecl_errors.so"
        rt
    )
    set_target_properties(sgf-kobuki-json-bridge PROPERTIES
        BUILD_RPATH "${KOBUKI_INSTALL_PREFIX}/lib"
    )
    target_link_options(sgf-kobuki-json-bridge PRIVATE "-Wl,--disable-new-dtags")
else()
    find_package(kobuki_core REQUIRED)
    target_link_libraries(sgf-kobuki-json-bridge PRIVATE kobuki_core::kobuki_core)
endif()

set_target_properties(sgf-kobuki-json-bridge PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin"
)
