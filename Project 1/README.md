# Simple Variational Monte Carlo solver for FYS4411

VMC-solver for project 1 of FYS4411 implemented in a class-structure written in C++. Datanalysis and plotting in Python. The article can be found in the subfolder named **Article**.

## Compiling and running the project
There are now several options you can use for compiling the project. If you use QT Creator, you can import this project into the IDE and point it to the `.pro`-file. If not, you can use CMake to create a Makefile for you which you can then run. You can install CMake through one of the Linux package managers, e.g., `apt install cmake`, `pacman -S cmake`, etc. For Mac you can install using `brew install cmake`. Other ways of installing are shown here: [https://cmake.org/install/](https://cmake.org/install/).

### Compiling the project using CMake
In a Linux/Mac terminal this can be done by the following commands
```bash
# Create build-directory
mkdir build

# Move into the build-directory
cd build

# Run CMake to create a Makefile
cmake ../

# Make the Makefile using two threads
make -j2

# Move the executable to the top-directory
mv vmc ..
```
Or, simply run the script `compile_project` via
```bash
./compile_project
```
and the same set of commands are done for you. Now the project can be run by executing
```bash
./vmc
```
in the top-directory.

#### Cleaning the directory
Run `make clean` in the top-directory to remove the executable `vmc` and the `build`-directory.
