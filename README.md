
# Data Parallel Library (dpl): An Extensible C++ SIMD Abstraction

A header-only C++ library for SIMD operations, designed with a focus on extensibility and zero-cost abstractions. This project serves as a showcase of modern C++ techniques (C++23) and hardware-level optimization. 

This library is an exploration of an idea I had about compile-time declarative dependency injection in C++. As you can probably tell from the state of the [`test`](test) folder, almost none of the code has been tested. It is currently optimized for Clang on Linux and is used as a playground for testing C++23 Modules and freestanding library design (By 'optimized for', I mean 'only works on').

## Why build this?

**Exploring C++23 and beyond:** I’m using this as a playground for new features.

**A Different Take on SIMD**: While I’m inspired by the STL’s `std::simd`, I’m not intending to replace the standard library. Instead, I aim to enable extensibility alongside portability. My goal was to design a SIMD library that leverages the user’s understanding of the underlying hardware—allowing generic code to be easily specialized for hardware-specific optimizations, and further extended when needed. Unlike `std::simd`, which is primarily designed as a portable abstraction, DPL can be used either as a portable abstraction or as a hardware-aware layer depending on the user’s preference and level of specialization.


## Technical Features

I started this project knowing how template-heavy the implementation was going to be; I chose to implement the project using C++20 to use concepts and constraints as well as C++ modules (though this was later changed to C++23 to access some better QoL features).

* **Extensibility:** Designed so you can inject custom logic, generator expressions, or backends without fighting the library’s internal structure (though you'd have to settle for fighting with my terrible naming sensibilities).
* **Freestanding Design:** By bypassing `std::`, the library has zero dependencies and is implemented entirely via compiler intrinsics. I understand some projects have very valid reasons for not using the standard library, e.g. bloat, freestanding requirement, ABI control, etc. I don’t really have a strong justification in that sense. I generally try to aim for minimalism over generalism; when first starting out, I didn't think I needed that many type-traits so I implemented it myself using compiler builtins. It worked out well in the beginning. A few iterations later, "a few" turned into most of the type-traits (and concepts) defined by the standard :see_no_evil:.
* **constexpr:** *Fixed-width* simd types should have all (including [math](include/dpl/core/math)) operations be constexpr


## Project Structure

* `include/dpl/std`: Standard library APIs
* `include/dpl/configuration`: Compiler/Target configuration definitions
* `include/dpl/preprocessor`: Helper preprocessor macros
* `include/dpl/core`: Core library headers
* `include/dpl/xmm`: Intel 128bit register implementations
* `modules/`: Interface modules
* `tests/`: Test files
* `tools/`: Build/debugger system scripts
* `docs/`: More specific documentation

## Running Tests
To run the tests, you'll need clang-21 (currently hard coded in the makefile) and GNU Make on Linux. It has not been tested on any other platforms outside of Linux.

```bash
make -j$(nproc) all
``` 
