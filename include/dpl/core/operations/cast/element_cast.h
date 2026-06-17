// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/array_for.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/constants/max_value.h"
#  include "dpl/core/immediate/constants/min_value.h"
#  include "dpl/core/immediate/constants/zero.h"
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
private:
    template <typename FromE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr ToE safe_cast(FromE val) noexcept {
        if consteval {
            // Do we need this?
            if constexpr (floating_point<FromE> && integral<ToE>) {
                auto const min = static_cast<FromE>(min_value_v<ToE>);
                auto const max = static_cast<FromE>(max_value_v<ToE>);
                auto const lt = val < min;
                auto const gt = val > max;
                if (lt || gt || !(val <= max && val >= min)) {
                    return dx::msb;
                }
            }
        }

        if constexpr (convertible_to<FromE, ToE>) {
            return static_cast<ToE>(val);
        } else {
            // TODO: Remove this and put it in the backend
            static_assert(floating_point<ToE> && floating_point<FromE>);
            static_assert(dx::digits_v<float> >= dx::digits_v<FromE>);
            return static_cast<ToE>(static_cast<float>(val));
        }
    }

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
                (Is < extent ? safe_cast(val[Is]) : dx::zero_v<ToE>)...};
            return dx::load<A>(aligned, buffer.data);
        }(val, iota_sequence<To>);
    }
};

template <typename ToE>
struct canonical_impl<element_cast_t<ToE>> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<remove_cvref_t<T>, ToE> operator()(
        T val) noexcept
    requires requires { element_cast<ToE>(internal::abi<T>, val); }
    {
        return element_cast<ToE>(internal::abi<T>, val);
    }
};

template <typename From, typename To>
concept rebound_element_castable =
    requires { typename rebind_simd_t<remove_cvref_t<From>, To>; } &&
    common_simd_type_with<rebind_simd_t<remove_cvref_t<From>, To>, From> &&
    explicitly_convertible_to<From, rebind_simd_t<remove_cvref_t<From>, To>>;

template <typename ToE>
struct extended_impl<element_cast_t<ToE>> {
    template <extended_vector T>
    requires simd_element_for<ToE, simd_abi_type_t<T>> &&
        rebound_element_castable<T, ToE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        using To = rebind_simd_t<remove_cvref_t<T>, ToE>;
        return static_cast<To>(__DPL forward<T>(val));
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
