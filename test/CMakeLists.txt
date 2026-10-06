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
        lexer.cpp
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

# Test the C runtime independently of the compiler's closure lowering.
find_package(Threads REQUIRED)
add_executable(mim-par-pool-test par_pool.c ${PROJECT_SOURCE_DIR}/src/mim/plug/ll/rt/mim_rt.c)
target_compile_features(mim-par-pool-test PRIVATE c_std_11)
target_link_libraries(mim-par-pool-test PRIVATE Threads::Threads)
add_dependencies(mim_all mim-par-pool-test)
foreach(THREADS 1 4)
    add_test(NAME par-pool-${THREADS} COMMAND mim-par-pool-test)
    set_tests_properties(par-pool-${THREADS} PROPERTIES ENVIRONMENT "MIM_NUM_THREADS=${THREADS}" TIMEOUT 30)
endforeach()

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
