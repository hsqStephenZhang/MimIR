pybind11_add_module(pymim
    driver.cpp
    world.cpp
    sym.cpp
    py.cpp
    ast.cpp
    # app.cpp
    log.cpp
    flags.cpp
    lam.cpp
    parser.cpp
    parser_wrapper.cpp
#    def_wrapper.cpp
    def.cpp
    error.cpp
)

set_target_properties(
    pymim
    PROPERTIES
        LIBRARY_OUTPUT_NAME _mim_core
        INSTALL_RPATH "$ORIGIN"
)
target_link_libraries(pymim PRIVATE libmim)
install(TARGETS pymim LIBRARY DESTINATION mim)
install(
    FILES
        ${PROJECT_SOURCE_DIR}/python/mim/__init__.py
        ${PROJECT_SOURCE_DIR}/python/mim/__init__.pyi
        ${PROJECT_SOURCE_DIR}/python/mim/_axioms.py
        ${PROJECT_SOURCE_DIR}/python/mim/_axioms.pyi
        ${PROJECT_SOURCE_DIR}/python/mim/_axioms_generated.py
        ${PROJECT_SOURCE_DIR}/python/mim/_mim_core.pyi
        ${PROJECT_SOURCE_DIR}/python/mim/py.typed
    DESTINATION mim
)
