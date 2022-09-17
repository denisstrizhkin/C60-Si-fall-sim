#!/bin/bash

# Stop script on error
set -e

# Check if dnf is present
if [ ! -z "$(which dnf)" ]; then
    is_dnf=1
fi

# Check if apt is present
if [ ! -z "$(which apt)" ]; then
    is_apt=1
fi

# Clone lammps repo
SRC_DIR="$HOME"/Desktop/src/lammps
if [ ! -d "$SRC_DIR" ]; then
    mkdir -p "$SRC_DIR"
    git clone -b release https://github.com/lammps/lammps.git "$SRC_DIR"
fi
cd "$SRC_DIR"
mkdir build
cd build

# Install required dnf packages
if [ "$is_dnf" -eq 1 ]; then
    if [ -z "$(dnf list installed | grep "^rpmfusion-free")" ]; then
        sudo dnf install https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
    fi

    packages=(
        cmake
        clang
        libomp-devel
        openmpi-devel
        ffmpeg
        voro++-devel
        python3-devel
    )

    sudo dnf install ${packages[*]}
fi

# Install required apt packages
if [ "$is_apt" -eq 1 ]; then
    packages=(
        cmake
        clang
        libomp-dev
        openmpi-dev
        ffmpeg
        voro++-dev
        python3-dev
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
)

cmake_flags=''
for build_option in "${build_options[@]}"
do
    cmake_flags="$cmake_flags -D ${build_option}=yes"
done

cmake ../cmake
cmake $cmake_flags .

# Compile and install
make -j10
make install
make install-python
