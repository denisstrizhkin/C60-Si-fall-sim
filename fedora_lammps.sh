#!/bin/bash

# Stop script on error
set -e

is_dnf=0
is_apt=0

# Check if dnf is present
if [ -n "$(which dnf)" ]; then
    is_dnf=1
fi

# Check if apt is present
if [ -n "$(which apt)" ]; then
    is_apt=1
fi

# Clone lammps repo
SRC_DIR=/mnt/data/lammps/lammps_repo
if [ ! -d "$SRC_DIR" ]; then
    mkdir -p "$SRC_DIR"
    git clone -b develop https://github.com/lammps/lammps.git "$SRC_DIR"
    mkdir build
fi
cd "$SRC_DIR"

if [ ! -d build ]; then
    mkdir build
fi
cd build

# Install required dnf packages
if [ "$is_dnf" -eq 1 ]; then
    if ! dnf list installed | grep -q "^rpmfusion-free"; then
        sudo dnf install \
        "https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm" \
        "https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm"
    fi

    packages=(
        cmake
        clang
        libomp-devel
        openmpi-devel
        ffmpeg
        python3-devel
    )

    sudo dnf install ${packages[*]}
fi

# Install required apt packages
if [ "$is_apt" -eq 1 ] && [ "$is_dnf" -eq 0 ]; then
    packages=(
        cmake
        clang
        libomp-dev
        libopenmpi-dev
        ffmpeg
        python3-dev
        python3-venv
    )

    sudo apt install ${packages[*]}
fi

# Set openmpi lib path
export MPI_HOME=/usr/lib64/openmpi

# Set cmake flags
build_options=(
    BUILD_MPI
    PKG_OPENMP
    PKG_BODY
    PKG_MANYBODY
    PKG_MISC
    PKG_VORONOI
    PKG_EXTRA-FIX
    PKG_EXTRA-COMPUTE
    BUILD_SHARED_LIBS
    LAMMPS_EXCEPTIONS
    PKG_PYTHON
    PKG_GPU
)

cmake_flags=''
for build_option in "${build_options[@]}"
do
    cmake_flags="$cmake_flags -D ${build_option}=yes"
done

if lspci | grep -qE '(VGA|3D).*NVIDIA'; then
  cmake_flags="$cmake_flags -D GPU_API=cuda -D GPU_ARCH=sm_80"
fi

cmake $cmake_flags ../cmake

# Compile and install
cmake --build . -j "$(nproc)"
cmake --install .