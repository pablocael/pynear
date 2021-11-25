# vptree-cpp
 C++ efficient Vantage Point Tree Implementation with Python 3 bindings.
 
 PS: this project is a Work in Progress

# Install pyvptree
python setup.py install


# Using C++ library
You can install vptree C++ library header using cmake. The library is a single header only.

To install, run:

```console
mkdir build
cd build

cmake ..
make
make install
```

# Development

The C++ code is a one header file within include/VPTree.hpp.

To use the project with some compiling tools on can use CMake to export the compile commands:

```console
mkdir build
cd build

cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=1 ../
```

