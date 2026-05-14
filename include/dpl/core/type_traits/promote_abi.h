// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_class.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

/**
 * @brief SIMD ABI promotion trait for fixed-width ABI types.
 *
 * `promote_abi` defines a structural *successor relation* over SIMD ABI types
 * in a fixed-width ABI hierarchy (e.g. xmm → ymm → zmm).
 *
 * It models ABI widening for operations that require a strictly larger
 * register configuration, such as concatenation or explicit vector expansion.
 *
 * @par User Specialization Model
 * Users MUST specialize `promote_abi` only for *fixed-width ABI types*.
 *
 * The `fixed_width_class<T>` specialization exists purely for internal
 * normalization and convenience, and is NOT part of the user-facing
 * specialization interface.
 *
 * In other words:
 * - User-facing specializations operate on ABI types directly (e.g. xmm, ymm).
 * - Wrapper types are automatically unwrapped before promotion is applied.
 *
 * @par Semantics
 * For a valid fixed-width ABI `A`, `promote_abi<A>::type` must satisfy:
 *
 * - It is a valid SIMD ABI type.
 * - It represents a strictly wider register configuration than `A`.
 * - It is the canonical next ABI in the fixed-width ABI hierarchy.
 *
 * @par Width Invariant
 * Promotion is strictly monotonic with respect to ABI size:
 *
 *     promote_abi<A>::type::size > A::size
 *
 * where `A` is a fixed-width ABI type.
 *
 * @par Wrapper Resolution (Internal Only)
 * If `A` is a `fixed_width_class<T>`, it is transparently resolved as:
 *
 *     promote_abi<A> := promote_abi<typename A::abi_type>
 *
 * This mechanism is implementation detail and not part of the user API.
 *
 * @par Design Constraints
 * - Only fixed-width ABI types are valid specialization targets.
 * - The ABI set must form a strictly increasing size-ordered chain.
 * - Each ABI must have a unique successor ABI.
 * - Promotion must terminate at a maximal ABI (e.g. zmm-class ABI).
 *
 * @par Non-goals
 * This trait does NOT apply to:
 * - Scalable/vector-length ABIs (e.g. SVE, RVV)
 * - Runtime-variable-width SIMD models
 * - ABI policy selection (see `common_abi`)
 *
 * @par Relationship to Other Traits
 * - `promote_abi`: structural ABI widening (fixed-width ABI lattice)
 * - `common_abi`: execution ABI selection (compatibility join)
 *
 * These traits are orthogonal:
 * - `promote_abi` defines representation growth for fixed-width ABIs only.
 * - `common_abi` defines execution compatibility across operands.
 *
 * @see promote_abi_t
 * @see common_abi
 */
DPL_EXPORT template <typename>
struct promote_abi {};

DPL_EXPORT template <fixed_width_class T>
struct promote_abi<T> : promote_abi<typename T::abi_type> {};

DPL_EXPORT template <typename A>
requires (fixed_width_class<A> || simd_abi<A>) && requires {
    typename promote_abi<A>::type;
    requires simd_abi<typename promote_abi<A>::type>;
    requires promote_abi<A>::type::size > internal::abi_width<A>();
}
using promote_abi_t DPL_NODEBUG = typename promote_abi<A>::type;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
