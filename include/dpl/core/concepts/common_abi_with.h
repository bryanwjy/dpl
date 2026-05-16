// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_class.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

/**
 * @brief Variadic SIMD ABI compatibility and selection trait.
 *
 * `common_abi<Ts...>` computes a single SIMD ABI type that is compatible
 * with all input ABI types in `Ts...`.
 *
 * It serves as the ABI "join" operation used for:
 * - binary and variadic SIMD operations
 * - expression result ABI selection
 * - implicit ABI unification across operands
 *
 * @par Input domain
 * Each `Ts` must be a valid SIMD type or SIMD ABI type:
 * - `simd_class<T>` (SIMD value or mask wrapper types)
 * - `simd_abi<T>` (core ABI types)
 *
 * @par User specialization rules
 * Users MUST only specialize `common_abi` for *pair-wise ABI combinations*
 * of `simd_abi` types.
 *
 * Variadic specializations are not required and are typically derived
 * from pair-wise rules.
 *
 * @par Convenience wrappers
 * Specializations involving `simd_class<T>` are provided purely for
 * convenience and ergonomic usage.
 *
 * These wrapper-based overloads:
 * - are not part of the core ABI specialization surface
 * - are forwarded internally to `simd_abi`-based specializations
 * - should not be specialized by users
 *
 * @par Semantics (fixed-width ABI domain)
 * If all input ABIs are fixed-width:
 *
 * - The result is the least-upper-bound (LUB) under ABI width ordering.
 * - Equivalently, the result is the smallest ABI whose width is
 *   greater than or equal to the maximum input ABI width.
 *
 *     width(result) = min { W | W ≥ max(width(Ts...)) }
 *
 * This corresponds to standard SIMD promotion behavior (e.g. xmm → ymm → zmm).
 *
 * @par Semantics (scalable ABI domain)
 * If any input ABI is a scalable ABI:
 *
 * - The result is implementation-defined.
 * - The result must be a valid SIMD ABI type.
 * - The result must be capable of representing all inputs under its
 *   execution semantics (including masking/predication rules if applicable).
 * - The result must be deterministic within a given ABI configuration.
 *
 * No ordering relation is assumed between scalable and fixed-width ABIs.
 *
 * @par Mixed ABI domain rule
 * For any mixture of fixed-width and scalable ABIs:
 *
 * - The result ABI is implementation-defined.
 * - It must satisfy the scalability domain invariants of the implementation.
 *
 * @par Invariants
 * Regardless of ABI family:
 *
 * - (Validity) The result is a valid SIMD ABI type.
 * - (Compatibility) All inputs are representable under the result ABI.
 * - (Determinism) The result is consistent for a given ABI configuration.
 * - (Commutative) `common_abi` is pair-wise commutative:
 *      common_abi<A, B>::type = common_abi<B, A>::type
 *
 * For fixed-width-only inputs:
 * - (Monotonicity) result width is ≥ maximum input width.
 *
 * @par Non-goals
 * `common_abi` does NOT:
 * - define ABI promotion chains (see `promote_abi`)
 * - define instruction selection or codegen strategy
 * - enforce representation optimality
 *
 * @par Relationship to other traits
 * - `promote_abi`: structural ABI widening within fixed-width lattice
 * - `common_abi`: compatibility join across ABI operands
 *
 * These traits are orthogonal:
 * - `promote_abi` is a unary lattice successor operation.
 * - `common_abi` is a variadic join operation.
 *
 * @see common_abi_t
 * @see promote_abi
 */
DPL_EXPORT template <typename... Ts>
struct common_abi {};

namespace internal {
template <typename T>
concept common_abi_arg = (simd_class<T> || simd_abi<T>);
}

DPL_EXPORT template <internal::common_abi_arg... Ts>
using common_abi_t = typename common_abi<Ts...>::type;

DPL_EXPORT template <simd_abi A>
struct common_abi<A> {
    using type DPL_NODEBUG = A;
};

DPL_EXPORT template <simd_class T>
struct common_abi<T> {
    using type DPL_NODEBUG = typename T::abi_type;
};

DPL_EXPORT template <simd_class T>
struct common_abi<T, T> {
    using type DPL_NODEBUG = typename T::abi_type;
};

DPL_EXPORT template <simd_abi A>
struct common_abi<A, A> {
    using type DPL_NODEBUG = A;
};

DPL_EXPORT template <simd_class A, simd_abi B>
struct common_abi<A, B> : common_abi<typename A::abi_type, B> {};
DPL_EXPORT template <simd_abi A, simd_class B>
struct common_abi<A, B> : common_abi<A, typename B::abi_type> {};
DPL_EXPORT template <simd_class A, simd_class B>
struct common_abi<A, B> :
    common_abi<typename A::abi_type, typename B::abi_type> {};

DPL_EXPORT template <typename T, typename U, typename... Ts>
requires requires { typename common_abi<T, U>::type; }
struct common_abi<T, U, Ts...> : common_abi<common_abi_t<T, U>, Ts...> {};

namespace atom {
template <typename A, typename B>
concept monotonic_common_abi =
    (scalable_abi<common_abi_t<A, B>> || scalable_abi<A> || scalable_abi<B>) ||
    (common_abi_t<A, B>::size >= A::size &&
        common_abi_t<A, B>::size >= B::size);

template <typename A, typename B>
concept common_abi_with = requires {
    typename common_abi_t<A, B>;
    typename common_abi_t<B, A>;
    requires same_as<common_abi_t<A, B>, common_abi_t<B, A>> &&
        simd_abi<common_abi_t<A, B>> && simd_abi<common_abi_t<B, A>>;
    requires monotonic_common_abi<A, B> && monotonic_common_abi<B, A>;
};
} // namespace atom

DPL_EXPORT template <typename A, typename B>
concept common_abi_with =
    simd_abi<A> && simd_abi<B> && atom::common_abi_with<A, B>;

DPL_EXPORT template <typename A, typename B>
concept same_abi_as = common_abi_with<A, B> && same_as<A, B>;

DPL_EXPORT template <typename A, typename B>
concept common_abi_simd_with = simd_class<A> && simd_class<B> &&
    atom::common_abi_with<typename A::abi_type, typename B::abi_type>;

DPL_EXPORT template <typename A, typename B>
concept same_abi_simd_as = simd_class<A> && simd_class<B> &&
    same_abi_as<typename A::abi_type, typename B::abi_type>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
