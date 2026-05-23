// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwand.h"
#include "dpl/core/operations/logical/result.h"

#if !DPL_MODULES
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void logical_and(...) noexcept = delete;

struct logical_and_t;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_logical_and = requires(L lhs, R rhs) {
    {
        logical_and(internal::abi<A>, lhs, rhs)
    } -> canonical_logical_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_logical_and = requires(L lhs, R rhs) {
    { logical_and(lhs, rhs) } -> extended_logical_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_logical_and = unqualified_canonical_logical_and<L, R, A> ||
    unqualified_extended_logical_and<L, R, A> ||
    (decayable_mask_for<L, operation_category::lane_agnostic> &&
        decayable_mask_for<R, operation_category::lane_agnostic> &&
        regular_invocable<logical_and_t, canonical_type_t<L>,
            canonical_type_t<R>>);

struct logical_and_t {
private:
    template <typename LE, typename RE, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<LE, A> lhs, basic_mask<RE, A> rhs) noexcept {
        return dx::bwand(lhs, rhs);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires common_size_with<LE, RE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<RE, A> operator()(
        basic_mask<LE, A> lhs, basic_mask<RE, A> rhs) noexcept {
        if constexpr (unqualified_canonical_logical_and<basic_mask<LE, A>,
                          basic_mask<RE, A>>) {
            if consteval {
                return fallback(lhs, rhs);
            } else {
                return logical_and(internal::abi<A>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_abi LA, common_abi_with<LA> RA, typename LE,
        common_size_with<LE> RE>
    requires simd_element_for<LE, LA> && simd_element_for<RE, RA> &&
        (scalable_abi<LA> || scalable_abi<RA> || different_from<LA, RA>) &&
        unqualified_canonical_logical_and<basic_mask<LE, LA>,
            basic_mask<RE, RA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<RE, common_abi_t<LA, RA>> operator()(
        basic_mask<LE, LA> lhs, basic_mask<RE, RA> rhs) noexcept {
        return logical_and(internal::abi<common_abi_t<LA, RA>>, lhs, rhs);
    }

    template <simd_mask L, simd_mask R>
    requires (extended_mask<L> || extended_mask<R>) &&
        unqualified_logical_and<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_logical_and<L, R>) {
            return logical_and(lhs, rhs);
        } else {
            return operator()(dx::to_canonical(lhs), dx::to_canonical(rhs));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::logical_and_t logical_and{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
