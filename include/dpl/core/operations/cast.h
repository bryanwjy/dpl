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
#  include "dpl/core/type_traits/basic_element.h"
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
    template <basic_simd_element From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To safe_cast(From val) noexcept {
        if consteval {
            // Do we need this?
            if constexpr (floating_point<From> && integral<To>) {
                auto const lt = val < min_value_v<To>;
                auto const gt = val > max_value_v<To>;
                if (lt || gt ||
                    !(val <= max_value_v<To> && val >= min_value_v<To>)) {
                    return dx::msb;
                }
            }
        }

        return static_cast<To>(val);
    }

    template <basic_simd_element From, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<To, A> fallback(
        basic_simd<From, A> arg) noexcept {
        using S = basic_simd<From, A>;
        using R = basic_simd<To, A>;
        return []<size_t... Is>(S arg, index_sequence<Is...>) {
            constexpr auto extent = element_count<S> < element_count<R>
                ? element_count<S>
                : element_count<R>;
            array_for<R> buffer{
                (Is < extent ? safe_cast(arg[Is]) : dx::zero_v<To>)...};
            return dx::load<R>(aligned, buffer.data);
        }(arg, iota_sequence<R>);
    }

public:
    template <simd_type From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(From mask) noexcept {
        if constexpr (unqualified_element_castable_to<From, To>) {
            if constexpr (basic_simd_type<From>) {
                if consteval {
                    return fallback(mask);
                } else {
                    return cast<To>(internal::abi<From>, mask);
                }
            } else {
                return cast<To>(internal::abi<From>, mask);
            }
        } else if constexpr (basic_simd_type<From>) {
            return fallback(mask);
        } else {
            return operator()(dx::to_basic_type(mask));
        }
    }
};

template <simd_element To>
requires (!basic_simd_element<To>)
struct cast_t<To> {
public:
    template <simd_type From>
    requires unqualified_element_castable_to<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From arg) noexcept {
        return cast<To>(internal::abi<From>, arg);
    }

    template <simd_type From>
    requires (!unqualified_element_castable_to<From, To>) && requires {
        typename basic_element_t<To>;
        typename invoke_result_t<cast_t<basic_element_t<To>>,
            basic_type_t<From>>;
        requires explicitly_convertible_to<
            invoke_result_t<cast_t<basic_element_t<To>>, basic_type_t<From>>,
            rebind_simd_t<From, To>>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(From arg) noexcept {
        using result = rebind_simd_t<From, To>;
        using base = cast_t<basic_element_t<To>>;
        return static_cast<result>(base::operator()(dx::to_basic_type(arg)));
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
    requires (!unqualified_castable_to<From, To>) && requires {
        typename basic_type_t<To>;
        typename invoke_result_t<cast_t<basic_type_t<To>>, From>;
        requires explicitly_convertible_to<
            invoke_result_t<cast_t<basic_type_t<To>>, From>, To>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To operator()(From arg) noexcept {
        using base = cast_t<basic_type_t<To>>;
        return static_cast<To>(base::operator()(arg));
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
