// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/array_for.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/constants/max_value.h"
#  include "dpl/core/immediate/constants/min_value.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename>
void element_cast(...) noexcept = delete;

template <typename E>
struct element_cast_t :
    private cast_operation_base<element_cast_t<E>>,
    private maskable_transform_base<element_cast_t<E>> {
    static_assert(is_object_v<E> && !is_const_v<E> && !is_volatile_v<E>);
    using operation_base<element_cast_t<E>>::operator();
    using maskable_transform_base<element_cast_t<E>>::operator();
};

template <typename ToE>
struct operation_signature<element_cast_t<ToE>> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename ToE>
struct fallback_impl<element_cast_t<ToE>> {
public:
    template <fixed_width_abi A>
    requires simd_element_for<ToE, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(basic_vector<ToE, A> val) noexcept {
        return val;
    }

    template <fixed_width_abi A, simd_element_for<A> FromE>
    requires simd_element_for<ToE, A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<ToE, A>
        DPL_VECTORCALL operator()(basic_vector<FromE, A> val) noexcept {
        using From = basic_vector<FromE, A>;
        using To = basic_vector<ToE, A>;
        return []<size_t... Is>(From val, index_sequence<Is...>) {
            constexpr auto extent =
                simd_abi_traits<From>::size < simd_abi_traits<To>::size
                ? simd_abi_traits<From>::size
                : simd_abi_traits<To>::size;
            array_for<To> buffer{
                (Is < extent ? static_cast<ToE>(val[Is]) : dx::zero_v<ToE>)...};
            return dx::load<A>(aligned, buffer.data);
        }(val, iota_sequence<To>);
    }
};

template <typename ToE, typename S, typename M, typename T>
concept unqualified_canonical_melement_cast =
    cpo_invocable<element_cast_t<ToE>, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<element_cast_t<ToE>, T>>) &&
    requires(S src, M mask, T val) {
        {
            element_cast<ToE>(
                internal::abi<cpo_result_t<element_cast_t<ToE>, T>>, src, mask,
                val)
        } -> same_as<cpo_result_t<element_cast_t<ToE>, T>>;
    };

template <typename ToE>
struct canonical_impl<element_cast_t<ToE>> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using result_t DPL_NODEBUG = basic_vector<ToE, simd_abi_type_t<T>>;

public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(T val) noexcept
    requires requires { element_cast<ToE>(internal::abi<T>, val); }
    {
        return element_cast<ToE>(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires canonical_vector<T> &&
        unqualified_canonical_melement_cast<ToE, result_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, mask_t<T> mask, T val) noexcept {
        return element_cast<ToE>(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_melement_cast<ToE, result_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, M cmask, T val) noexcept {
        return element_cast<ToE>(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_melement_cast<ToE, dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return element_cast<ToE>(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_melement_cast<ToE, dx::zero_t,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, M cmask, T val) noexcept {
        return element_cast<ToE>(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename To, typename From>
concept unqualified_extended_element_cast =
    simd_element_for<To, simd_abi_type_t<From>> &&
    requires { typename rebind_simd_t<remove_cvref_t<From>, To>; } &&
    common_simd_type_with<rebind_simd_t<remove_cvref_t<From>, To>, From> &&
    (explicitly_convertible_to<From, rebind_simd_t<remove_cvref_t<From>, To>> ||
        requires {
            {
                element_cast<To>(internal::declarg<From>())
            } -> same_as<rebind_simd_t<remove_cvref_t<From>, To>>;
        });

template <typename ToE, typename S, typename M, typename T>
concept unqualified_extended_melement_cast =
    cpo_invocable<element_cast_t<ToE>, T> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<element_cast_t<ToE>, T>>) &&
    requires {
        {
            element_cast<ToE>(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<element_cast_t<ToE>, T>>;
    };

template <typename ToE>
struct extended_impl<element_cast_t<ToE>> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <extended_vector T>
    requires unqualified_extended_element_cast<ToE, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        using To = rebind_simd_t<remove_cvref_t<T>, ToE>;
        if constexpr (explicitly_convertible_to<T, To>) {
            return static_cast<To>(__DPL forward<T>(val));
        } else {
            return element_cast<ToE>(__DPL forward<T>(val));
        }
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_melement_cast<ToE, S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return element_cast<ToE>( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_melement_cast<ToE, S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return element_cast<ToE>( __DPL forward<S>(src),
            dx::to_const_mask<S>(cmask), __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<element_cast_t<ToE>, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_melement_cast<ToE, dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return element_cast<ToE>(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<element_cast_t<ToE>, T> M>
    requires unqualified_extended_melement_cast<ToE, dx::zero_t,
        launder_cmask_t<cpo_result_t<element_cast_t<ToE>, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return element_cast<ToE>(zero,
            dx::to_const_mask<cpo_result_t<element_cast_t<ToE>, T>>(cmask),
            __DPL forward<T>(val));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename To>
inline constexpr internal::element_cast_t<To> element_cast{};
} // namespace cpo
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
