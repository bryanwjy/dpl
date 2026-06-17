// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/cast/abi_cast.h"
#include "dpl/core/operations/cast/element_cast.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_canonical.h"
#  include "dpl/core/concepts/common_simd_type_with.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_volatile.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename>
void simd_cast(...) noexcept = delete;

template <typename T>
struct simd_cast_t : private cast_operation_base<simd_cast_t<T>> {
    static_assert(is_object_v<T> && !is_const_v<T> && !is_volatile_v<T>);
    using operation_base<simd_cast_t<T>>::operator();
};

template <typename T>
struct operation_signature<simd_cast_t<T>> {
    static consteval void operator()(simd_type auto&&) noexcept {}
};

template <simd_type To>
struct fallback_impl<simd_cast_t<To>> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, ALWAYS_INLINE)
    static constexpr To operator()(To&& val) noexcept {
        return __DPL forward<To>(val);
    }

    template <simd_type From>
    requires common_simd_type_with<From, To> &&
        cpo_invocable<abi_cast_t<simd_abi_type_t<To>>, From> &&
        same_as<cpo_result_t<abi_cast_t<simd_abi_type_t<To>>, From>, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From&& from) noexcept {
        return dx::abi_cast<simd_abi_type_t<To>>( __DPL forward<From>(from));
    }

    template <simd_type From>
    requires common_simd_type_with<From, To> &&
        cpo_invocable<element_cast_t<simd_element_type_t<To>>, From> &&
        same_as<cpo_result_t<element_cast_t<simd_element_type_t<To>>, From>, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From from) noexcept {
        return dx::element_cast<simd_element_type_t<To>>(from);
    }
};

template <typename From, typename To>
concept unqualified_canonical_castable_to =
    common_simd_type_with<From, To> && requires(From from) {
        { simd_cast<To>(internal::abi<From>, from) } -> same_as<To>;
    };

template <typename From, typename To>
concept unqualified_canonical_castable_from =
    common_simd_type_with<From, To> && requires(From from) {
        { simd_cast<To>(internal::abi<To>, from) } -> same_as<To>;
    };

template <typename From, typename To>
concept unqualified_canonical_simd_cast =
    unqualified_canonical_castable_to<From, To> ||
    unqualified_canonical_castable_from<From, To>;

template <simd_type To>
struct canonical_impl<simd_cast_t<To>> {
    template <canonical_simd_type From>
    requires (extended_simd_type<To> && constructible_from<To, From>) ||
        unqualified_canonical_simd_cast<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, ALWAYS_INLINE)
    static constexpr To operator()(From val) noexcept {
        if constexpr (extended_simd_type<To> && constructible_from<To, From>) {
            return To(val);
        } else if constexpr (unqualified_canonical_castable_to<From, To>) {
            return simd_cast<To>(internal::abi<From>, val);
        } else {
            return simd_cast<To>(internal::abi<To>, val);
        }
    }
};

template <simd_type To>
struct extended_impl<simd_cast_t<To>> {
    template <extended_simd_type From>
    requires common_simd_type_with<From, To> &&
        (same_as<canonical_type_t<From>, To> ||
            explicitly_convertible_to<From, To>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD, ALWAYS_INLINE)
    static constexpr To operator()(From&& val) {
        if constexpr (same_as<canonical_type_t<From>, To>) {
            return dx::to_canonical(__DPL forward<From>(val));
        } else {
            return static_cast<To>(__DPL forward<From>(val));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename To>
inline constexpr internal::simd_cast_t<To> simd_cast{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
