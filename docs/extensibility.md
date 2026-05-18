
# Extending the Data Parallel Library (DPL)

> This documentation was written by the author and refined with the assistance of AI-based editing tools for clarity and readability.

I am not an expert in SIMD backends or in the full range of applications for hardware intrinsics beyond relatively simple linear algebra on small vectors and matrices. There are users with far deeper expertise in specific architectures and instruction sets, and DPL is designed with that in mind.

A core design goal of DPL is extensibility through modern C++ idioms. By leveraging various C++ features, the library provides a robust contract for extending SIMD functionality while maintaining zero-cost abstractions. Extensions are not limited to custom backends—they also include expression templates and specialized optimizations that integrate directly into the library’s dispatch logic.


## Custom Backends

In this section, we will walk through how to define custom backends, enabling DPL to integrate with unfamiliar or specialized SIMD hardware.

SIMD backends (also referred to as `abi_type` or `abi_tag` in the library) are used by DPL to select and bind a concrete ABI-specific implementation to a SIMD type. They act as the mechanism through which the library maps high-level SIMD abstractions onto hardware-specific vector and mask representations.

The primary templates, `basic_vector<E, A>` and `basic_mask<E, A>` (where `E` is the element type), take the backend `A` as their second template parameter. This parameter is used to specialize the SIMD types so that they encapsulate the underlying vector and mask types expected by the target hardware intrinsics.

In DPL, SIMD ABIs are required to satisfy a **base** set of constraints:

```c++
template<typename A>
concept simd_abi = enable_simd_abi<A> && is_empty_v<A> && semiregular<A> && requires {
    typename integral_constant<A, A{}>;
    requires /*is-unary-template*/<A::template native_vector>;
    requires /*is-unary-template*/<A::template native_mask>;
};
```

The concept `is-unary-template` is exposition-only. To model `simd_abi`, a type must specialize `dpl::datapar::enable_simd_abi` to evaluate to true, and satisfy all additional constraints listed above.

Once a type satisfies `simd_abi`, it is further classified as either a _fixed-width_ or _scalable_ ABI. Every `simd_abi` **must model exactly one of these categories**.

```c++
template <typename T>
concept fixed_width_abi = simd_abi<T> && requires {
    typename integral_constant<size_t, T::size>;
} && (T::size != scalable_size);


template <typename T>
concept scalable_abi = !fixed_width_abi<T> && simd_abi<T> &&
    (T::size == scalable_size) && requires {
        /* has-size-function-template */
    };
```

> Note: The API and constraints for `scalable_abi` are still under development. While DPL itself may evolve, support for scalable backends is currently more experimental. In particular, the design reflects my limited practical experience with scalable SIMD architectures, and may change as those use cases are better understood.

The concept `has-size-function-template`, is an exposition-only concept that is satisfied if the class has a static member function template that returns the number of SIMD lanes given a simd-element. In practice, this concept cannot be defined, instead the implementation only checks for `requires { T::size<char>() } -> unsigned_integral`.

A simple example is the ABI for the 128-bit xmm registers on the x86 architecture, which is implemented in the `dpl.xmm` module found [here](../include/dpl/xmm/basic/abi.h).

However, defining a backend alone is not sufficient. DPL requires a minimal set of operations to be provided so that it can construct and manipulate SIMD types associated with the ABI.

The following operations must be defined:

- `broadcast<E>(abi_type, scalar)`  
  Broadcasts a scalar value into a SIMD object.

- `broadcast<E>(abi_type, bool)`  
  Broadcasts a boolean value into a SIMD mask.

- `extract(abi_type, simd, idx)`  
  Extracts the scalar value at the given index from a SIMD object.

- `extract(abi_type, mask, idx)`  
  Extracts the boolean value at the given index from a SIMD mask.

- `initialize<E>(abi_type, scalars...)`  
  Constructs a SIMD object from a sequence of scalar values.

- `load(abi_type, scalar_ptr)`  
  Loads a sequence of scalar values from memory into a SIMD object.

- `store(abi_type, simd, dst_ptr)`  
  Stores a SIMD object into memory.

All of these functions are discovered via argument-dependent lookup (ADL). See the [*Customization Point Objects*](#customization-point-objects) section for details on how dispatch is performed.

Fallback implementations are provided for higher-level operations, ranging from arithmetic to algorithmic primitives. Many of these do not directly use hardware intrinsics; instead, they are expressed in terms of the basic operations listed above and rely on the compiler to generate efficient code. As a result, performance may be suboptimal depending on the operation and target architecture.

For performance-critical use cases, users are encouraged to provide custom implementations of the SIMD operations they require. As with the core operations, these are discovered via argument-dependent lookup (ADL). Provided that the underlying basic operations are `constexpr`-compatible, the fallback implementations will also evaluate in a `constexpr` context. This does not imply SIMD execution at compile time; rather, during `constexpr` evaluation, SIMD objects are treated as fixed-size arrays.

In order for this to work, the compiler must know the number of elements at compile time so that it can both iterate over them and construct the resulting object. All fallback implementations are written with this assumption. As a consequence, fallbacks for scalable ABIs are not implementable in this model, since their size is not known at compile time. This also suggests that operations on scalable ABIs cannot, in general, be evaluated in a `constexpr` context.

It is also worth noting that a `simd_abi` is not required to map directly to hardware SIMD types or instructions. A custom backend may instead use arrays and conventional loops. This can be useful for testing, experimentation, or as a reference implementation.


## Customization Point Objects (CPOs)

[Customization Point Objects (CPOs)](https://en.cppreference.com/cpp/named_req/CustomizationPointObject) are semantically invoked like functions but are implemented as constant stateless function objects to provide a robust, predictable interface for library extensions. For a deep dive what they are, I highly recommend Barry Revzin’s [Customization Points article](https://brevzin.github.io/c++/2020/12/19/cpo-niebloid/) and the [std::ranges documentation](https://en.cppreference.com/cpp/ranges).

> Personal Note: My interest in this pattern grew while developing [r20](https://github.com/bryanwjy/r20), a small library focused on backporting C++23 Ranges features to C++20.

In `std::ranges` specifically, many of the CPOs are written to either look for member functions (like `x.begin()`) or valid unqualified lookups (`begin(x)`), DPL on the other hand, currently _only_ uses it for unqualified lookups. In DPL, almost all public SIMD operations are exposed as CPOs. This ensures that the entire library remains open for user extension.

The primary advantage here is the declarative approach. Users can define their own specialized functions in their own namespaces; as long as the DPL CPOs are invoked in a context where the compiler has visibility of those declarations, "it just works."

This provides several benefits:

* **Low Friction:** Users do not need to perform complicated engineering work to "hook" into the library or register types.
* **Open/Closed Principle:** Extensions are handled through standard language rules, meaning users never have to modify the core dpl namespace.

In DPL, every SIMD class must define an `abi_type` member. This type typically serves as the anchor for argument-dependent lookup (ADL).

When invoking a DPL operation, dispatch follows this pattern:

```c++
struct foo_t { // simplified CPO
template<simd_vector T>
static auto operator()(T lhs, T rhs) noexcept {
    constexpr typename T::abi_type abi{};
    if constexpr (requires { foo(abi, lhs, rhs); }) {
        return foo(abi, lhs, rhs); // Unqualified call triggers ADL
    } else {
        return dpl_fallback(lhs, rhs);
    }
}
};
```

Because ADL considers the namespaces of the argument types, passing `abi_type` as the leading argument ensures that `foo(abi, ...)` is discovered in the namespace associated with the abi/backend. This makes the customization mechanism robust across different namespaces and modules.

The trade-off is that users must design their overloads carefully: since lookup is driven by ADL, poorly constrained or conflicting overloads can lead to ambiguity.

> **Caveat:** When providing custom implementations of any operation for a backend, if all argument types are the primary DPL SIMD types (`basic_vector` or `basic_mask`), the return type must also be the corresponding primary DPL type. Many parts of DPL rely on this invariant, and returning non-primary types (e.g., expression templates) in these cases may lead to incorrect behavior or ill-formed programs.

## Custom Simd Types

In some cases, users may choose not to use the primary SIMD templates provided by DPL—for example, due to differing coding styles, conflicting abstractions, or the need for tighter control over representations and operations.

DPL generalizes not only over element types and ABI types, but also over SIMD and mask types themselves. Both SIMD and mask types are modeled through the `simd_class` abstraction and are subject to their own sets of constraints:

```c++
template <typename T>
concept simd_basics = /*exposition-only*/
    is_object_v<T> && semiregular<T> && is_trivially_copyable_v<T> && requires {
        typename T::value_type;
        typename T::abi_type;
        requires simd_abi<typename T::abi_type>;
    };

template <typename T>
concept simd_vector = /*simd-basics*/<T> && enable_simd_type<T> &&
    requires {
        requires simd_element_for<typename T::value_type, typename T::abi_type>;
        requires sizeof(typename T::value_type) <= T::abi_type::size;
    } &&
    explicitly_convertible_to<T,
        typename T::abi_type::template native_vector<typename T::value_type>>;

template <typename M>
concept simd_mask = /*simd-basics*/<T> && enable_simd_mask<M> && requires {
    typename M::simd_vector;
    requires same_as<typename M::value_type, bool> &&
        simd_vector<typename M::simd_vector>;
    requires explicitly_convertible_to<M,
        typename M::abi_type::template native_mask<
            typename M::simd_vector::value_type>>;
};
```

As with `simd_abi`, these types are further classified into *fixed-width* and *scalable* variants.

Custom SIMD and mask types must be convertible to their corresponding native types. This conversion serves as a bridge within DPL, allowing generic components to interoperate with user-defined types.

In particular, DPL provides the utility function `dpl::datapar::to_canonical`, which converts a custom SIMD or mask type into the corresponding primary DPL type. This enables fallback behavior: if a custom type does not provide specialized implementations for certain operations, DPL can transparently fall back to the default implementations defined for the primary templates.

Aside from the basic operations, higher-level operations follow a similar dispatch strategy:

- If the type is a custom SIMD type:
  - Attempt unqualified lookup (ADL) for the operation.
  - If a matching overload is found, use it.
  - Otherwise, convert the value to the corresponding basic DPL type via `to_canonical` and retry.

- If the type is a primary DPL SIMD type:
  - Attempt unqualified lookup (ADL) for the operation.
  - If a matching overload is found, use it.
  - Otherwise, fall back to the default DPL implementation.

For _scalable_ backends, no default DPL fallback is provided. As a result, programs may be ill-formed if the backend does not supply an implementation for a required operation.

## Expressions Templates

There is one additional motivation for defining custom SIMD types that was not covered in the previous section. DPL’s extensible design—through custom SIMD types and CPO-based ADL dispatch—enables users to implement expression templates that integrate seamlessly with the library. Expression templates represent computations as compile-time expression trees, allowing deferred evaluation, eliminating intermediate temporaries, and enabling more efficient, backend-specific code generation.

Depending on compiler options/capability, SIMD operations may be evaluated eagerly, which may introduce unnecessary intermediate temporaries and multiple passes over the data. For example, an expression such as:

```c++
auto c = a + b * d;
```

may be evaluated as a sequence of independent operations, producing intermediate results that are stored and reloaded. This can lead to suboptimal code generation, increased register pressure, and unnecessary memory traffic.

Expression templates allows implementations to reason about the full operation at once, enabling optimizations such as operation fusion, elimination of temporaries, and more efficient use of hardware intrinsics.

Within DPL, this approach is particularly powerful when combined with custom SIMD types. Through CPO-based ADL dispatch, users can intercept and specialize how expressions are constructed and evaluated, allowing them to tailor execution to specific hardware or application requirements without modifying the core library.

For example, consider a custom SIMD type that returns the following struct from its multiplication operation:

```c++
struct multiply {
    using value_type = float;
    using abi_type = xmm::abi_tag;

    __m128 lhs;
    __m128 rhs;

    explicit operator __m128(this multiply self) noexcept {
        return _mm_mul_ps(self.lhs, self.rhs);
    }

    template<simd_with<float, abi_type> R>
    friend basic_vector<float, abi_type> add(abi_type, multiply lhs, R rhs) noexcept {
        return _mm_fmadd_ps(lhs.lhs,lhs.rhs, static_cast<__m128>(rhs));
    }

    template<simd_with<float, abi_type> L>
    friend basic_vector<float, abi_type> add(abi_type, L lhs, multiply rhs) noexcept {
        return _mm_fmadd_ps(rhs.lhs, rhs.rhs, static_cast<__m128>(lhs));
    }
};
```

A subsequent `add` operation will resolve to one of the overloads above via ADL, allowing the implementation to perform a fused multiply-add instead of evaluating the multiplication and addition as separate operations.

DPL provides the necessary mechanisms to support expression templates, but barely defines any in the core library. More advanced compositions—such as fused operations (e.g., multiply-add)—are intentionally left to user-defined implementations, allowing backends to exploit hardware-specific capabilities where appropriate.

One such example in the core library is logical negation of the primary SIMD mask type (`operator!`), which returns an unexported [`negated_mask`](../include/dpl/core/operations/negated_mask.h) type. This type participates in subsequent operations (e.g., `select`) by altering their behavior—for instance, swapping arguments—so that the negation need not be materialized explicitly. This is the *only* case where an expression template is used in DPL.

### Limitations

Operations involving the primary DPL SIMD types (`basic_vector` and `basic_mask`) must always return primary types. Returning expression template types from such operations violates this expectation and may lead to incorrect behavior or ill-formed programs, as many components of DPL rely on this invariant.

Users wishing to implement expression templates should do so in conjunction with custom SIMD and mask types. This ensures that all participating operations are consistently defined and that DPL’s dispatch and fallback mechanisms behave as intended.
