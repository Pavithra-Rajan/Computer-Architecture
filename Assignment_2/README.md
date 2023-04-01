# Assignment 2 CS4032D Computer Architecture
The assignment entails comparing the performance of cache based on hit time, miss rate and average memory access time by varying the cache size of L1 and L2.
L1 = 16 KB, 32 KB, 64 KB
L2 = 2 MB, 4 MB, 8 MB

## Installation and set-up
```console
sudo apt-get install build-essential git m4 scons zlib1g zlib1g-dev libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev python-dev python
```

```console
git clone https://gem5.googlesource.com/public/gem5
```

```console
cd gem5
scons build/<configuration>/gem5.opt -j <number of CPUs>
```
As my system us x86, I will replace <configuration> with ```X86```. You can find the configuration and the number of CPUs by running the lscpu command.
```opt``` indicates debug and optimisation. ```debug``` can be used during debgugging gem5 code without any optimisation. Other binaries to describe the optimisation level are fast, perf and prof.

## Run gem5
```console
build/X86/gem5.opt configs/example/se.py -c tests/test-progs/hello/bin/x86/linux/hello
```
This is for running the simulation of a simple hello world program.

The output files are saved in the ```m5out``` directory. The files present are:
- ```stats.txt```: These are the gem5 simulation statistics.
- ```config.ini```: The simulated machine configuration in gem5
- ```config.json```: This is the same as config.ini but in jsonified format

# Analysing gem5 output
- sim_seconds: how long the program ran i the simulated machine
- host seconds: how long it took in the host machine

# Directory details
```console
tree - L 1
```
```console
.
├── build
├── build_opts
├── build_tools
├── CODE-OF-CONDUCT.md
├── configs
├── CONTRIBUTING.md
├── COPYING
├── ext
├── include
├── KCONFIG.md
├── LICENSE
├── m5out
├── MAINTAINERS.yaml
├── pyproject.toml
├── README
├── RELEASE-NOTES.md
├── requirements.txt
├── SConsopts
├── SConstruct
├── site_scons
├── src
├── system
├── TESTING.md
├── tests
└── util

12 directories, 13 files
```
- build - formed after running the scond build command
- build_opts - holds files that define default settings for build of different configurations
- configs - contains simulation configurations
- ext - contains gem5 dependencies which are external to gem5
- include - contains include files for compiling gem5
- m5out - all the output results can be found here
- src - contains gem5 source file 
- system - contains low level softwares like firmware or bootloaders in simulated system
- test - files related to gem5 regression tests
- util - contains utility scripts and programs
# gem5 
- gem5 is an amalgamation of m5 and GEMS. 

- m5 is developed by the University of Michigan with multiple CPU models and ISA. It also has classical memory model. CPU and memory in m5 interact with each other as master and slave using the available m5 ports.

- GEMS is built by University of Wisconsin with Ruby memory model.  It has difference coherence protocols. 

- gem5 is a modular discrete event driven simulator. Events are stored in event queue and each event has a timestamp associated with it. 

- It is a full system simulator as it can simualte both user and kernal code. (SE and FS mode)

- It can operate in the SE mode which is the SysCall Emulation mode. It emulates the system level code. 

- gem5 simulates a machine on the host machine - simulated machine

- It takes a simulation script written in python as input and the simulated machine's behaviour is written in C++.

- Can simulate two CPU models - in-order and out of order

- Can have two memory models - classic and ruby memory model
