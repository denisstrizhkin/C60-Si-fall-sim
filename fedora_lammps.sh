#!/bin/bash

SRC_DIR="$HOME"/Desktop/src/lammps
if [ ! -d "$SRC_DIR" ]; then
    mkdir -p "$SRC_DIR"
    git clone -b release https://github.com/lammps/lammps.git "$SRC_DIR"
fi
cd "$SRC_DIR"
mkdir build
cd build

# Install rpmfusion repo
if [ -z "$(dnf list installed | grep "^rpmfusion-free")" ];
then
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

export MPI_HOME=/usr/lib64/openmpi

build_options=(
    BUILD_MPI
    BUILD_OMP
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

make -j10
make install