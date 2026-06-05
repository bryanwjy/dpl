// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/bwnot.h"
#include "dpl/core/operations/logical/result.h"
#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void logical_not(...) noexcept = delete;

struct logical_not_t;

template <typename T, typename A = typename T::abi_type>
concept unqualified_canonical_logical_not = requires(T val) {
    { logical_not(internal::abi<A>, val) } -> canonical_logical_result<T, T, A>;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_extended_logical_not = requires(T val) {
    { logical_not(val) } -> extended_logical_result<T, T, A>;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_logical_not = unqualified_extended_logical_not<T, A> ||
    (decayable_mask_for<T, operation_category::lane_agnostic> &&
        regular_invocable<logical_not_t, canonical_type_t<T>>);

struct logical_not_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_mask<E, A> val) noexcept {
        return dx::bwnot(val);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_logical_not<basic_mask<E, A>, A>) {
            if consteval {
                return fallback(val);
            } else {
                return logical_not(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_logical_not<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        basic_mask<E, A> val) noexcept {
        return logical_not(internal::abi<A>, val);
    }

    template <extended_mask T>
    requires unqualified_logical_not<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) {
        if constexpr (unqualified_extended_logical_not<T>) {
            return logical_not(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::logical_not_t logical_not{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
