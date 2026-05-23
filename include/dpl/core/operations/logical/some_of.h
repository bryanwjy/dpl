// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/logical/all_of.h"
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
void some_of(...) noexcept = delete;

struct some_of_t;

template <typename T, typename A = typename T::abi_type>
concept unqualified_canonical_some_of = requires(T val) {
    { some_of(internal::abi<A>, val) } -> boolean_testable;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_extended_some_of = requires(T val) {
    { some_of(val) } -> boolean_testable;
};

template <typename T, typename A = typename T::abi_type>
concept unqualified_some_of =
    unqualified_canonical_some_of<T, A> || unqualified_extended_some_of<T, A> ||
    (decayable_mask_for<T, operation_category::lane_reduction> &&
        regular_invocable<some_of_t, canonical_type_t<T>>);

struct some_of_t {
private:
    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr bool DPL_VECTORCALL fallback(
        basic_mask<E, A> mask) noexcept {
        return dx::any_of(mask) && !dx::all_of(mask);
    }

public:
    template <fixed_width_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(basic_mask<E, A> val) noexcept {
        if constexpr (unqualified_canonical_some_of<basic_mask<E, A>, A>) {
            if consteval {
                return fallback(val);
            } else {
                return some_of(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <scalable_abi A, simd_element_for<A> E>
    requires unqualified_canonical_some_of<basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr bool operator()(basic_mask<E, A> val) noexcept {
        return some_of(internal::abi<A>, val);
    }

    template <extended_mask T>
    requires unqualified_some_of<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_some_of<T>) {
            return some_of(val);
        } else {
            return operator()(dx::to_canonical(val));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::some_of_t some_of{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
