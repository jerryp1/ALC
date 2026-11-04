"""Build the source extension for the active platform; no binaries are bundled."""

import shlex
import subprocess

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

try:
    flags = shlex.split(
        subprocess.check_output(
            ["pkg-config", "--cflags", "--libs", "libcrypto"], text=True
        )
    )
    libdir = subprocess.check_output(
        ["pkg-config", "--variable=libdir", "libcrypto"], text=True
    ).strip()
except (OSError, subprocess.CalledProcessError) as exc:
    raise RuntimeError(
        "Install a C++20 compiler, pkg-config and OpenSSL development headers; see README.md"
    ) from exc

include_dirs = [flag[2:] for flag in flags if flag.startswith("-I")]
library_dirs = [flag[2:] for flag in flags if flag.startswith("-L")]
libraries = [flag[2:] for flag in flags if flag.startswith("-l")]
link_args = [flag for flag in flags if not flag.startswith(("-I", "-L", "-l"))]
link_args += ["-pthread", "-Wl,-rpath," + libdir]
setup(
    ext_modules=[
        Pybind11Extension(
            "cpp.alc_tans_cpp_codec",
            ["src/cpp/serving_runtime.cpp"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            libraries=libraries,
            cxx_std=20,
            extra_compile_args=["-O3", "-pthread", "-fno-strict-aliasing"],
            extra_link_args=link_args,
        )
    ],
    cmdclass={"build_ext": build_ext},
)
