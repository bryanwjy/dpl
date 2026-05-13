// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/bitwise.h"

#if !DPL_MODULES
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/max_value.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
struct cast_t {};

template <typename>
void cast(...) noexcept = delete;

template <typename From, typename To>
concept unqualified_element_castable_to =
    simd_type<From> && simd_element<To> && requires(From arg) {
        {
            cast<To>(internal::abi<From>, arg)
        } -> simd_with<To, typename From::abi_type>;
    };

template <simd_element To>
struct cast_t<To> {
private:
    template <typename From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr To safe_cast(From val) noexcept {
        if consteval {
            // Do we need this?
            if constexpr (floating_point<From> && integral<To>) {
                auto const min = static_cast<From>(min_value_v<To>);
                auto const max = static_cast<From>(max_value_v<To>);
                auto const lt = val < min;
                auto const gt = val > max;
                if (lt || gt || !(val <= max && val >= min)) {
                    return dx::msb;
                }
            }
        }

        if constexpr (convertible_to<From, To>) {
            return static_cast<To>(val);
        } else {
            static_assert(floating_point<To> && floating_point<From>);
            static_assert(dx::digits_v<float> >= dx::digits_v<From>);
            return static_cast<To>(static_cast<float>(val));
        }
    }

    template <typename From, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<To, A> DPL_VECTORCALL
        fallback(basic_simd<From, A> arg) noexcept {
        using S = basic_simd<From, A>;
        using R = basic_simd<To, A>;
        return []<size_t... Is>(S arg, index_sequence<Is...>) {
            constexpr auto extent =
                simd_abi_traits<S>::size < simd_abi_traits<R>::size
                ? simd_abi_traits<S>::size
                : simd_abi_traits<R>::size;
            array_for<R> buffer{
                (Is < extent ? safe_cast(arg[Is]) : dx::zero_v<To>)...};
            return dx::load<R>(aligned, buffer.data);
        }(arg, iota_sequence<R>);
    }

    template <typename T>
    using basic_value DPL_NODEBUG = typename basic_type_t<T>::value_type;

public:
    template <fixed_width_simd From>
    requires basic_simd_type<rebind_simd_t<From, To>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        if constexpr (unqualified_element_castable_to<From, To>) {
            if constexpr (basic_simd_type<From>) {
                if consteval {
                    return fallback(val);
                } else {
                    return cast<To>(internal::abi<From>, val);
                }
            } else {
                return cast<To>(internal::abi<From>, val);
            }
        } else if constexpr (basic_simd_type<From>) {
            return fallback(val);
        } else {
            return operator()(dx::to_basic_type(val));
        }
    }

    template <scalable_simd From>
    requires basic_simd_type<rebind_simd_t<From, To>> &&
        (unqualified_element_castable_to<From, To> ||
            unqualified_element_castable_to<basic_type_t<From>, To>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From val) noexcept {
        if constexpr (unqualified_element_castable_to<From, To>) {
            return cast<To>(internal::abi<From>, val);
        } else {
            return cast<To>(internal::abi<From>, dx::to_basic_type(val));
        }
    }

    template <simd_type From>
    requires (!basic_simd_type<rebind_simd_t<From, To>>) &&
        unqualified_element_castable_to<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From arg) noexcept {
        return cast<To>(internal::abi<From>, arg);
    }

    template <simd_type From>
    requires (!basic_simd_type<rebind_simd_t<From, To>> &&
                 !unqualified_element_castable_to<From, To>) &&
        requires {
            typename basic_value<rebind_simd_t<From, To>>;
            typename invoke_result_t<
                cast_t<basic_value<rebind_simd_t<From, To>>>, From>;
            requires regular_invocable<reinterpret_t<rebind_simd_t<From, To>>,
                invoke_result_t<cast_t<basic_value<rebind_simd_t<From, To>>>,
                    From>>;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From arg) noexcept {
        using result = rebind_simd_t<From, To>;
        using base = cast_t<basic_value<result>>;
        return dx::reinterpret<result>(base::operator()(arg));
    }
};

template <typename From, typename To>
concept unqualified_castable_to =
    simd_type<From> && simd_type<To> && requires(From arg) {
        {
            cast<To>(internal::abi<common_abi_t<From, To>>, arg)
        } -> core_convertible_to<To>;
    };

template <simd_type To>
struct cast_t<To> {
public:
    template <simd_type From>
    requires same_abi_simd_as<To, From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From arg) noexcept {
        if constexpr (basic_simd_type<From>) {
            return cast_t<typename To::value_type>::operator()(arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }

    template <unqualified_castable_to<To> From>
    requires common_abi_simd_with<To, From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From arg) noexcept {
        return cast<To>(internal::abi<common_abi_t<From, To>>, arg);
    }
};

template <simd_type To>
requires (!basic_simd_type<To>)
struct cast_t<To> {
public:
    template <unqualified_castable_to<To> From>
    requires common_abi_simd_with<To, From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From arg) noexcept {
        return cast<To>(internal::abi<common_abi_t<From, To>>, arg);
    }

    template <simd_type From>
    requires (!unqualified_castable_to<From, To>) &&
        requires {
            typename basic_type_t<To>;
            typename invoke_result_t<cast_t<basic_type_t<To>>, From>;
        } &&
        (explicitly_convertible_to<
             invoke_result_t<cast_t<basic_type_t<To>>, From>, To> ||
            regular_invocable<reinterpret_t<To>,
                invoke_result_t<cast_t<basic_type_t<To>>, From>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From arg) noexcept {
        using base = cast_t<basic_type_t<To>>;
        if constexpr (explicitly_convertible_to<
                          invoke_result_t<cast_t<basic_type_t<To>>, From>,
                          To>) {
            return static_cast<To>(base::operator()(arg));
        } else {
            return dx::reinterpret<To>(base::operator()(arg));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename To>
inline constexpr internal::cast_t<To> cast{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
