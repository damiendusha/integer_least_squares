#!/bin/bash

mkdir build
mkdir bin

cd build

cmake -DCMAKE_INSTALL_PREFIX=../ ..

make -j6
make install
