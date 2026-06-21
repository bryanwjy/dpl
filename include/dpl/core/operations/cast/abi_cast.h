// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/array_for.h"

#if !DPL_MODULES
#  include "dpl/core/basic/from_bitset.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/store.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/basic/to_native_type.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_native_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename>
void abi_cast(...) noexcept = delete;
void abi_cast(...) noexcept = delete;
template <typename A>
struct abi_cast_t : private cast_operation_base<abi_cast_t<A>> {
    static_assert(is_object_v<A> && !is_const_v<A> && !is_volatile_v<A>);
    using operation_base<abi_cast_t<A>>::operator();
};

template <typename A>
struct operation_signature<abi_cast_t<A>> {
    static consteval void operator()(simd_type auto&&) noexcept {}
};

template <simd_abi ToA>
struct fallback_impl<abi_cast_t<ToA>> {
    template <simd_element_for<ToA> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, ToA> operator()(
        basic_vector<E, ToA> val) noexcept {
        return val;
    }

    template <simd_element_for<ToA> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_mask<E, ToA> operator()(
        basic_mask<E, ToA> val) noexcept {
        return val;
    }

    template <canonical_simd_type T>
    requires common_abi_with<simd_abi_type_t<T>, ToA> &&
        simd_element_for<simd_element_type_t<T>, ToA> &&
        same_as<simd_native_type_t<T>,
            simd_native_type_t<rebind_simd_t<T, simd_element_type_t<T>, ToA>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<T, simd_element_type_t<T>, ToA> operator()(
        T val) noexcept {
        return dx::to_native_type(val);
    }

    template <fixed_width_abi FromA, simd_element_for<FromA> E>
    requires simd_element_for<E, ToA> && fixed_width_abi<ToA> &&
        common_abi_with<ToA, FromA> &&
        (!same_as<simd_native_type_t<basic_vector<E, ToA>>,
            simd_native_type_t<basic_vector<E, FromA>>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, ToA>
        DPL_VECTORCALL operator()(basic_vector<E, FromA> val) noexcept {
        using From = basic_vector<E, FromA>;
        using To = basic_vector<E, ToA>;
        if constexpr (From::size > To::size) {
            array_for<From> buffer DPL_INDETERMINATE;
            dx::store(val, buffer.data);
            return dx::load<To>(buffer.data);
        } else {
            array_for<To> buffer DPL_INDETERMINATE;
            dx::store(val, buffer.data);
            return dx::load<To>(buffer.data);
        }
    }

    template <fixed_width_abi FromA, simd_element_for<FromA> E>
    requires simd_element_for<E, ToA> && fixed_width_abi<ToA> &&
        common_abi_with<ToA, FromA> &&
        (!same_as<simd_native_type_t<basic_mask<E, ToA>>,
            simd_native_type_t<basic_mask<E, FromA>>>) &&
        cpo_invocable<to_bitset_t, basic_mask<E, FromA>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<E, ToA>
        DPL_VECTORCALL operator()(basic_mask<E, FromA> val) noexcept {
        using From = basic_mask<E, FromA>;
        using To = basic_mask<E, ToA>;
        if constexpr (From::size != To::size) {
            auto const bits = dx::to_bitset(val);
            return dx::from_bitset<To>(static_cast<bitset<To::size>>(bits));
        } else {
            return dx::from_bitset<To>(dx::to_bitset(val));
        }
    }
};
template <typename From, typename To>
concept unqualified_canonical_vector_target_acastable =
    simd_vector<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        {
            abi_cast<To>(internal::abi<From>, arg)
        } -> vector_with<simd_element_type_t<From>, To>;
    };

template <typename From, typename To>
concept unqualified_canonical_vector_source_acastable =
    simd_vector<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        {
            abi_cast(internal::abi<To>, arg)
        } -> vector_with<simd_element_type_t<From>, To>;
    };

template <typename From, typename To>
concept unqualified_canonical_mask_target_acastable =
    simd_mask<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        {
            abi_cast<To>(internal::abi<From>, arg)
        } -> mask_with<simd_element_type_t<From>, To>;
    };

template <typename From, typename To>
concept unqualified_canonical_mask_source_acastable =
    simd_mask<From> && simd_abi<To> &&
    simd_element_for<simd_element_type_t<From>, To> && requires(From arg) {
        {
            abi_cast(internal::abi<To>, arg)
        } -> mask_with<simd_element_type_t<From>, To>;
    };

template <typename From, typename To>
concept unqualified_canonical_source_abi_cast =
    unqualified_canonical_vector_source_acastable<From, To> ||
    unqualified_canonical_mask_source_acastable<From, To>;

template <typename From, typename To>
concept unqualified_canonical_target_abi_cast =
    unqualified_canonical_vector_target_acastable<From, To> ||
    unqualified_canonical_mask_target_acastable<From, To>;

template <typename From, typename To>
concept unqualified_canonical_abi_cast =
    unqualified_canonical_source_abi_cast<From, To> ||
    unqualified_canonical_target_abi_cast<From, To>;

template <simd_abi ToA>
struct canonical_impl<abi_cast_t<ToA>> {
    template <canonical_simd_type From>
    requires common_abi_with<simd_abi_type_t<From>, ToA> &&
        unqualified_canonical_abi_cast<From, ToA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<From, simd_abi_type_t<From>, ToA> operator()(
        From val) noexcept {
        if constexpr (unqualified_canonical_target_abi_cast<From, ToA>) {
            return abi_cast<ToA>(internal::abi<From>, val);
        } else {
            // ToA extends FromA
            return abi_cast(internal::abi<ToA>, val);
        }
    }
};

template <typename From, typename To>
concept rebound_abi_castable =
    requires {
        typename rebind_simd_t<remove_cvref_t<From>, simd_element_type_t<From>,
            To>;
    } &&
    common_simd_type_with<
        rebind_simd_t<remove_cvref_t<From>, simd_element_type_t<From>, To>,
        From> &&
    explicitly_convertible_to<From,
        rebind_simd_t<remove_cvref_t<From>, simd_element_type_t<From>, To>>;

template <simd_abi ToA>
struct extended_impl<abi_cast_t<ToA>> {
    template <extended_simd_type From>
    requires common_abi_with<simd_abi_type_t<From>, ToA> &&
        rebound_abi_castable<From, ToA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From&& val) {
        using To =
            rebind_simd_t<remove_cvref_t<From>, simd_element_type_t<From>, ToA>;
        return static_cast<To>(__DPL forward<From>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename To>
inline constexpr internal::abi_cast_t<To> abi_cast{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
