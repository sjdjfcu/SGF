if(NOT DEFINED JSON11_SOURCE_DIR)
    message(FATAL_ERROR "JSON11_SOURCE_DIR is not set")
endif()

set(json11_cpp "${JSON11_SOURCE_DIR}/json11.cpp")
if(NOT EXISTS "${json11_cpp}")
    message(FATAL_ERROR "json11.cpp not found at ${json11_cpp}")
endif()

file(READ "${json11_cpp}" json11_source)
string(FIND "${json11_source}" "#include <cstdint>" cstdint_include)

if(cstdint_include EQUAL -1)
    string(REPLACE
        "#include <cstdio>\n#include <limits>"
        "#include <cstdio>\n#include <cstdint>\n#include <limits>"
        patched_json11_source
        "${json11_source}"
    )

    if(patched_json11_source STREQUAL json11_source)
        message(FATAL_ERROR "Could not locate json11 include block to patch")
    endif()

    file(WRITE "${json11_cpp}" "${patched_json11_source}")
endif()
