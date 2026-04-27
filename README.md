
# Data Parallel Library (dpl): An Extensible C++ SIMD Abstraction

A header-only C++ library for SIMD operations, designed with a focus on extensibility and zero-cost abstractions. This project serves as a showcase of modern C++ techniques (C++23) and hardware-level optimization.

> Project Status: Experimental / Technical Portfolio
This library is an exploration of "clean-room" C++ development. It is currently optimized for Clang on Linux and is used as a playground for testing C++23 Modules and freestanding library design. (By 'optimized for', I mean 'only works on')

## Why build this?

**Curiosity & Free Time:** I wanted to see if I could build a working SIMD abstraction from the ground up

**Skill Showcase:** It’s a place for me to practice and display advanced C++ techniques—useful for when I eventually look for my next role.

**Exploring C++23:** I’m using this as a playground for new features like C++ Modules, Explicit Object Parameters (deducing this), and consteval logic.

**A Different Take on SIMD:** While I’m inspired by the STL’s `std::experimental::simd`, I wanted to prioritize extensibility over pure portability. I wanted to design a SIMD library that is able to leverage the user's understanding of the hardware, allowing for generic code that can be easily specialized for hardware-specific optimizations when needed.


## Technical Features

I started this project knowing how template-heavy the implementation was going to be; consequently, I chose to implement the project using C++20 to use concepts and constraints as well as C++ modules (though this was later changed to C++23 to access some better QoL features).

* **Extensibility:** Designed so you can inject custom logic, generator expressions, or backends without fighting the library’s internal structure (though you'd have to settle for fighting with my terrible naming sensibilities).
* **Freestanding Design:** By bypassing `std::`, the library has zero dependencies and is implemented entirely via compiler intrinsics. I understand some projects have very valid reasons for not using the standard library, e.g. bloat, freestanding requirement, ABI control, etc. I don't have a good reason. I always try to aim for minimalism over generalism. When I first started out I didn't think I needed that many type-traits so I implemented it myself using compiler builtins. It ended up requiring most of the type_traits and concepts header.


## Project Structure

* `include/dpl/std`: Standard library APIs
* `include/dpl/configuration`: Compiler/Target configuration definitions
* `include/dpl/preprocessor`: Helper preprocessor macros
* `include/dpl/core`: Core library headers
* `include/dpl/xmm`: Intel 128bit register implementations
* `modules/`: Interface modules
* `tests/`: Test files
* `tools/`: Build/debugger system scripts

## Running Tests
To run the tests, you'll need clang-21 (currently hard coded in the makefile) and GNU Make on Linux. It has not been tested on any other platforms outside of Linux.

```bash
make -j$(nproc) all
``` 
