# Each doctest executable needs exactly one TU instantiating doctest; generating it keeps that
# boilerplate out of the test sources - especially for auto-discovered third-party plugin tests.
set(MIM_DOCTEST_MAIN "${CMAKE_CURRENT_BINARY_DIR}/doctest_main.cpp")
file(GENERATE
    OUTPUT "${MIM_DOCTEST_MAIN}"
    CONTENT "#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN\n#include <doctest/doctest.h>\n"
)

function(add_mim_test)
    set(TEST ${ARGV0})

    cmake_parse_arguments(
        PARSE_ARGV 1        # skip first arg
        PARSED              # prefix of output variables
        ""                  # options (none)
        ""                  # one-value keywords (none)
        "SOURCES;PRIVATE"   # multi-value keywords
    )

    add_executable(${TEST})
    target_sources(${TEST}
        PRIVATE
            "${MIM_DOCTEST_MAIN}"
            ${PARSED_SOURCES}
    )
    target_link_libraries(${TEST}
        PRIVATE
            ${PARSED_PRIVATE}
            doctest::doctest
            libmim
            # No shared object to load, so a test only reaches its plugins by linking them in.
            $<$<BOOL:${MIM_STATIC_PLUGINS}>:mim_static_plugins>
    )
    add_dependencies(${TEST} mim_all_plugins)
    add_dependencies(mim_all ${TEST})
    doctest_discover_tests(${TEST})

    list(APPEND MIM_TEST_TARGETS ${TEST})
    set(MIM_TEST_TARGETS "${MIM_TEST_TARGETS}" CACHE INTERNAL "MIM_TEST_TARGETS")
endfunction()

set(MIM_TEST_TARGETS "" CACHE INTERNAL "MIM_TEST_TARGETS" FORCE)

add_mim_test(mim-test
    SOURCES
        cast.cpp
        dump.cpp
        graph_pattern.cpp
        level.cpp
        lexer.cpp
        nom.cpp
        pattern.cpp
        restricted_dep_types.cpp
        test.cpp
        variant.cpp
)

add_mim_test(mim-regex-test
    SOURCES
        automaton.cpp
    PRIVATE
        automaton
)

add_mim_test(mim-tensor-pattern-test
    SOURCES
        tensor_pattern.cpp
)
# Dynamic plugins are MODULE libraries and cannot be linked into a test. Compile the small
# read-only adapter into this executable; phases are exercised through the loaded plugin.
if(NOT MIM_STATIC_PLUGINS)
    target_sources(mim-tensor-pattern-test PRIVATE ../src/mim/plug/tensor/pattern.cpp)
endif()

mim_extra_plugins(test MIM_EXTRA_PLUGINS)
foreach(PLUGIN IN LISTS MIM_EXTRA_PLUGINS)
    set(DIR "${PROJECT_SOURCE_DIR}/extra/${PLUGIN}")
    file(GLOB SOURCES CONFIGURE_DEPENDS "${DIR}/test/*.cpp")
    if(NOT SOURCES)
        continue()
    endif()

    message(STATUS "Adding third-party Mim unit tests 'mim-${PLUGIN}-test' from 'extra/${PLUGIN}/test'")
    add_mim_test(mim-${PLUGIN}-test SOURCES ${SOURCES})

    if(EXISTS "${DIR}/include")
        target_include_directories(mim-${PLUGIN}-test PRIVATE "${DIR}/include")
    endif()
endforeach()
