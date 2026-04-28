
* Define a `scalar` abi
* Design an API where we can have vector of vectors, e.g. 256bit vector consisting of 2 independent 128 bit vectors whose elements cannot cross between one another unless the vectors are split into two 'real' 128 bit vectors and spliced.
* Investigate: Is there a better way to define a fallback for constexpr?
* Maybe rename `basic_simd` to `basic_vector` and `basic_simd_mask` to `basic_vector_mask`? So that there isn't name collisions with the traits and concepts?
* Make build system more configurable
  * e.g. different targets, different build config
* Expose a configuration to disable the ALWAYS_INLINE attribute
* How to default the simd ABI?
  * Should this be configurable?
* Support scalable vector ABIs
  * ~~How to deal with constexpr~~ Up to implementation, but typically disabled
* Maybe redesign `cload` API
* Define extensible concatenation API
  * i.e. How can one define a new ABI, e.g. ymm, that could then hook into DPL's concat mechanism
    such that something like `concat(xmm_vectora, xmm_vectorb)` is well-formed
  * How to deal with more complex cases, e.g. concat(xmm, ymm, xmm) => zmm
* Define where-expression API
* How to deal with non-IEEE floats?
