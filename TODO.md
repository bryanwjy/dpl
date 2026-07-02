
* Define a `scalar` abi
* Design an API where we can have vector of vectors, e.g. 256bit vector consisting of 2 independent 128 bit vectors whose elements cannot cross between one another unless the vectors are split into two 'real' 128 bit vectors and spliced.
* Investigate: Is there a better way to define a fallback for constexpr?
* Make build system more configurable
  * e.g. different targets, different build config
* Investigate GCC modules build
* Expose a configuration to disable the ALWAYS_INLINE attribute
* How to default the simd ABI?
  * Should this be configurable?
* Support scalable vector ABIs
  * ~~How to deal with constexpr~~ Up to implementation, but typically disabled
* Define where-expression API
* How to deal with non-IEEE floats?
