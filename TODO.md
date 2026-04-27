
* How to default the simd ABI?
  * Should this be configurable?
* Support scalable vector ABIs
  * ~~How to deal with constexpr~~ Up to implementation, but typically disabled
* Redesign `cload` API
* Define extensible concatenation API
  * i.e. How can one define a new ABI, e.g. ymm, that could then hook into DPL's concat mechanism
    such that something like `concat(xmm_vectora, xmm_vectorb)` is well-formed
  * How to deal with more complex cases, e.g. concat(xmm, ymm, xmm) => zmm
* Define where-expression API
* How to deal with non-IEEE floats?
