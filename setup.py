#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.

import glob
import os
import shutil
from os import path
from setuptools import setup
from typing import List
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from pathlib import Path

def get_extensions():
    extensions_dir = path.join("detectron2", "layers", "csrc")

    main_source = path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(path.join(extensions_dir, "**", "*.cpp"))

    from torch.utils.cpp_extension import ROCM_HOME

    is_rocm_pytorch = (
        True if ((torch.version.hip is not None) and (ROCM_HOME is not None)) else False
    )

    # common code between cuda and rocm platforms, for hipify version [1,0,0] and later.
    source_cuda = glob.glob(path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        path.join(extensions_dir, "*.cu")
    )
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if (
        torch.cuda.is_available() and ((CUDA_HOME is not None) or is_rocm_pytorch)
    ) or os.getenv("FORCE_CUDA", "0") == "1":
        extension = CUDAExtension
        sources += source_cuda

        if not is_rocm_pytorch:
            define_macros += [("WITH_CUDA", None)]
            extra_compile_args["nvcc"] = [
                "-O3",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]
        else:
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["nvcc"] = []

        nvcc_flags_env = os.getenv("NVCC_FLAGS", "")
        if nvcc_flags_env != "":
            extra_compile_args["nvcc"].extend(nvcc_flags_env.split(" "))

    include_dirs = [extensions_dir]

    ext_modules = [
        extension(
            "detectron2._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

    print("Sources to compile =", sources)
    print("include_dirs =", include_dirs)
    print("macros =", define_macros)
    print("extra_compile_args =", extra_compile_args)
    return ext_modules


def get_model_zoo_configs() -> List[str]:
    """
    Return a list of configs to include in package for model zoo. Copy over these configs inside
    detectron2/model_zoo.
    """

    # Use absolute paths while symlinking.
    source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
    destination = path.join(
        path.dirname(path.realpath(__file__)), "detectron2", "model_zoo", "configs"
    )
    # Symlink the config directory inside package to have a cleaner pip install.

    # Remove stale symlink/directory from a previous build.
    if path.exists(source_configs_dir):
        if path.islink(destination):
            os.unlink(destination)
        elif path.isdir(destination):
            shutil.rmtree(destination)

    if not path.exists(destination):
        try:
            os.symlink(source_configs_dir, destination)
        except OSError:
            # Fall back to copying if symlink fails: ex. on Windows.
            shutil.copytree(source_configs_dir, destination)

    config_paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
        "configs/**/*.py", recursive=True
    )
    return config_paths


# For projects that are relative small and provide features that are very close
# to detectron2's core functionalities, we install them under detectron2.projects
PROJECTS = {
    "detectron2.projects.point_rend": "projects/PointRend/point_rend",
    "detectron2.projects.deeplab": "projects/DeepLab/deeplab",
    "detectron2.projects.panoptic_deeplab": "projects/Panoptic-DeepLab/panoptic_deeplab",
}

setup(
    package_data={"detectron2.model_zoo": get_model_zoo_configs()},
    ext_modules=get_extensions(),
    include_package_data=False
)
