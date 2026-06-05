// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/enable_simd_abi.h"
#include "dpl/core/type_traits/simd_abi_type.h"

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
 * - `simd_type<T>` (SIMD value or mask wrapper types)
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
 * Specializations involving `simd_type<T>` are provided purely for
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
DPL_EXPORT template <typename... T>
struct common_abi {};

namespace internal {
template <typename T>
concept none_abi_type =
    requires { typename simd_abi_type_t<T>; } && !enable_simd_abi<T>;
} // namespace internal

DPL_EXPORT template <typename... Ts>
using common_abi_t = typename common_abi<Ts...>::type;

DPL_EXPORT template <typename T>
struct common_abi<T> : simd_abi_type<T> {};

DPL_EXPORT template <typename T>
struct common_abi<T, T> : common_abi<T> {};
DPL_EXPORT template <internal::none_abi_type A, internal::none_abi_type B>
struct common_abi<A, B> : common_abi<simd_abi_type_t<A>, simd_abi_type_t<B>> {};
DPL_EXPORT template <typename A, internal::none_abi_type B>
struct common_abi<A, B> : common_abi<A, simd_abi_type_t<B>> {};
DPL_EXPORT template <internal::none_abi_type A, typename B>
struct common_abi<A, B> : common_abi<simd_abi_type_t<A>, B> {};

DPL_EXPORT template <typename T, typename U, typename... Ts>
requires requires { typename common_abi_t<T, U>; }
struct common_abi<T, U, Ts...> : common_abi<common_abi_t<T, U>, Ts...> {};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
