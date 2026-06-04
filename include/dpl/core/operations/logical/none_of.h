// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/logical/any_of.h"

#if !DPL_MODULES
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/decayable.h"
#  include "dpl/core/concepts/operation_category.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/std/concepts/boolean_testable.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void none_of(...) noexcept = delete;

struct none_of_t;

template <typename T, typename A = typename T::abi_type>
concept unqualified_canonical_none_of = requires(T val) {
    { none_of(internal::abi<A>, val) } -> boolean_testable;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_extended_none_of = requires(T val) {
    { none_of(val) } -> boolean_testable;
};

template <typename T>
concept expression_none_of = mask_expression<T> &&
    regular_invocable<none_of_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_none_of =
    decayable_mask_for<T, operation_category::lane_reduction> &&
    regular_invocable<none_of_t, canonical_type_t<T>>;

template <typename T, typename A = typename T::abi_type>
concept extended_none_of = unqualified_extended_none_of<T, A> ||
    expression_none_of<T> || decayable_none_of<T>;

struct none_of_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL fallback(
        basic_mask<E, A> mask) noexcept {
        return !dx::any_of(mask);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_none_of<basic_mask<E, A>, A>) {
            if consteval {
                return fallback(val);
            } else {
                return none_of(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_none_of<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(basic_mask<E, A> val) noexcept {
        return none_of(internal::abi<A>, val);
    }

    template <extended_mask T>
    requires extended_none_of<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) {
        if constexpr (unqualified_extended_none_of<T>) {
            return none_of(val);
        } else if constexpr (expression_none_of<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::none_of_t none_of{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
