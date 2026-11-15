import os, subprocess, sys
sys.path.insert(0, os.path.dirname(__file__))
from mim_sh_test import MimShTest
from mim_roundtrip_test import MimRoundTripTest

config.name = 'mim regression'
# lit 23 deprecates the external shell, but our RUN lines rely on `$?` and `VAR=`, which the
# internal shell does not implement.
try:
    config.test_format = MimShTest(True, force_execute_external=True)
except TypeError:
    config.test_format = MimShTest(True)

config.suffixes = ['.mim']

if not getattr(config, 'test_source_root', None):
    config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.my_obj_root, 'lit', 'out')

# `--param roundtrip=mim|ast` ignores the RUN lines and checks instead that each input's `-o`/`--output-ast` output round-trips.
roundtrip = lit_config.params.get('roundtrip')
if roundtrip:
    # Optimizing twice is not idempotent - but the dump has to be.
    flags = {'mim': ['--no-opt', '--output-mim'], 'ast': ['--output-ast']}
    if roundtrip not in flags:
        lit_config.fatal(f"unknown roundtrip mode '{roundtrip}'; expected one of: {', '.join(flags)}")
    config.name = f'mim roundtrip-{roundtrip}'
    config.test_format = MimRoundTripTest(*flags[roundtrip])
    config.test_exec_root = os.path.join(config.my_obj_root, 'roundtrip', roundtrip)
    # `UNSUPPORTED: roundtrip` opts out an input that only compiles with flags from its RUN lines.
    config.available_features.add('roundtrip')

# `--param mim=<executable>` substitutes another compiler, e.g. a wrapper that adds flags to select a reference path.
mim = lit_config.params.get('mim', config.mim)
# `--param tile_law=0` realizes every `tile_req` by TileFuse's per-level recursion instead of a law where one expresses it.
tile_law = lit_config.params.get('tile_law')
if tile_law is not None:
    # A wrapper keeps `%mim` one executable path, as the scripts that run it expect.
    wrapper = os.path.join(config.my_obj_root, f'mim-tile-law-{tile_law}')
    with open(wrapper, 'w') as script:
        script.write(f'#!/bin/sh\nexec "{mim}" -X tensor:tile-law={tile_law} "$@"\n')
    os.chmod(wrapper, 0o755)
    mim = wrapper
    if tile_law == '0':
        config.available_features.add('tile-fuse-path')
config.substitutions.append(('%mim', mim))
config.substitutions.append(('%FileCheck', '"{}"'.format(config.filecheck)))
# Parent of the plugin directory, so that a test can spell a plugin as the path `mim/<plugin>`.
# os.path.join would mix in a backslash on Windows, which the external bash then swallows as an escape.
config.substitutions.append(('%libdir', config.my_obj_root + '/lib'))

# inherit env vars
config.environment = os.environ

config.available_features.add("always")

# Gate expensive tests on optimized builds via `// REQUIRES: release`.
if getattr(config, 'build_type', '').lower() in ('release', 'relwithdebinfo', 'minsizerel'):
    config.available_features.add("release")
# 16-bit floating-point support (needs std::float16_t) is detected once by CMake and passed in via
# lit.site.cfg.py. Tests emitting math.F16 literals gate on this via `// REQUIRES: fp16`.
if getattr(config, 'fp16', False):
    config.available_features.add("fp16")
# operating system is detected using sys.platform (see https://docs.python.org/3/library/sys.html#sys.platform)
# Tests that only work on e.g. Linux operating systems specify this via `// REQUIRES: system-linux`.
config.available_features.add(f"system-{sys.platform}")
# CUDA support (needs CUDA toolkit) is detected once by CMake and passed in via
# lit.site.cfg.py. Tests gate on this via `// REQUIRES: cuda`.
if getattr(config, 'cuda_can_run_on_device', False):
    config.available_features.add('cuda')
    # Code built for `sm_90a` runs only on a device of compute capability 9.0; tests run it via `%if cuda-sm90`.
    try:
        caps = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'],
                              capture_output=True, text=True, timeout=30).stdout.split()
        if caps and caps[0] == '9.0':
            config.available_features.add('cuda-sm90')
    except (OSError, subprocess.SubprocessError):
        pass
