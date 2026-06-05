// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/evaluate.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_expression.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/core/type_traits/simd_expression_result.h"
#  include "dpl/std/concepts/boolean_testable.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void any_of(...) noexcept = delete;

struct any_of_t;

template <typename T, typename A = typename T::abi_type>
concept unqualified_canonical_any_of = requires(T val) {
    { any_of(internal::abi<A>, val) } -> boolean_testable;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_extended_any_of = requires(T val) {
    { any_of(val) } -> boolean_testable;
};

template <typename T>
concept expression_any_of = mask_expression<T> &&
    regular_invocable<any_of_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_any_of =
    decayable_mask_for<T, operation_category::lane_reduction> &&
    regular_invocable<any_of_t, canonical_type_t<T>>;

template <typename T, typename A = typename T::abi_type>
concept extended_any_of = unqualified_extended_any_of<T, A> ||
    expression_any_of<T> || decayable_any_of<T>;

struct any_of_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL fallback(
        basic_mask<E, A> mask) noexcept {
        return []<size_t... Is>(auto mask, index_sequence<Is...>) {
            return (... || mask[Is]);
        }(mask, iota_sequence<E, A>);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_any_of<basic_mask<E, A>, A>) {
            if consteval {
                return fallback(val);
            } else {
                return any_of(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_any_of<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(basic_mask<E, A> val) noexcept {
        return any_of(internal::abi<A>, val);
    }

    template <extended_mask T>
    requires extended_any_of<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) {
        if constexpr (unqualified_extended_any_of<T>) {
            return any_of(val);
        } else if constexpr (expression_any_of<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::any_of_t any_of{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
