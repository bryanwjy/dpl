// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/abs.h"
#include "dpl/core/operations/bit.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/negate.h"
#include "dpl/core/operations/permute.h"
#include "dpl/core/operations/select.h"

#if !DPL_MODULES
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/basic/to_basic_type.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/constants/exponent_bias.h"
#  include "dpl/core/constants/exponent_bits.h"
#  include "dpl/core/constants/max_value.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/constants/zero.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/basic_element.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/to_integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
struct cast_t {};

template <basic_simd_element To>
struct cast_t<To> {
private:
    template <basic_simd_element From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr To safe_cast(From val) noexcept {
        if consteval {
            if constexpr (floating_point<From> && integral<To>) {
                if (val < min_value_v<To>) {
                    return min_value_v<To>;
                }

                if (val > max_value_v<To>) {
                    return max_value_v<To>;
                }
            }
        }

        return static_cast<To>(val);
    }

    template <simd_element From, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<To, A> safe_cast(
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

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_simd<to_signed_integral_t<E>, A> DPL_VECTORCALL
        ilogb(basic_simd<E, A> arg) noexcept {
        using int_type = to_signed_integral_t<E>;
        auto const biased = (arg & exponent_bits) >> imm<digits_v<E>>;
        return biased - static_cast<int_type>(exponent_bias_v<E>);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<To, A> DPL_VECTORCALL
        fallback(basic_simd<float, A> arg) noexcept
    requires integral<To>
    {
        if constexpr (common_size_with<int64, To>) {
            auto const parg = []<size_t... Is>(
                                  auto arg, index_sequence<Is...>) {
                return dx::permute<(Is / 2)...>(arg);
            }(arg, iota_sequence<float, A>);
            constexpr auto hidden_bit = 1 << dx::digits_v<float>;
            constexpr auto du64 = static_cast<uint64>(dx::digits_v<float>);
            auto const exp = ilogb(arg);
            auto const mantissa =
                dx::reinterpret<int32>(parg & dx::mantissa_bits) | hidden_bit;

            auto const uexp = dx::reinterpret<uint64>(exp) >> imm<32>;
            auto const umantissa = dx::reinterpret<uint64>(mantissa) >> imm<32>;
            auto const shift = dx::abs(uexp - du64);
            auto const large = dx::reinterpret<int64>(umantissa << shift);
            auto const small = dx::reinterpret<int64>(umantissa >> shift);

            constexpr auto signbits =
                dx::reinterpret<uint64>(dx::msb_v<basic_simd<int32, A>>);
            auto const is_lt_zero = dx::reinterpret<uint64>(parg) > signbits;
            if constexpr (signed_integral<To>) {
                auto const result = dx::bit_keep(
                    uexp > dx::zero, dx::select(uexp < du64, small, large));
                return dx::reinterpret<To>(dx::negate(is_lt_zero, result));
            } else {
                return dx::reinterpret<To>(
                    dx::bit_keep(dx::bwandnot(uexp > dx::zero, is_lt_zero),
                        dx::select(uexp < du64, small, large)));
            }
        } else if constexpr (common_size_with<int32, To>) {
            constexpr auto hidden_bit =
                dx::one_v<uint32> << dx::digits_v<float>;
            auto const exp = ilogb(arg);
            auto const umantissa =
                dx::reinterpret<uint32>(arg & dx::mantissa_bits) | hidden_bit;
            auto const shift = dx::abs(exp - dx::digits_v<float>);
            auto const large = dx::reinterpret<int32>(umantissa << shift);
            auto const small = dx::reinterpret<int32>(umantissa >> shift);
            auto const result =
                dx::select(exp < dx::digits_v<float>, small, large);
            if constexpr (signed_integral<To>) {
                return dx::reinterpret<To>(dx::negate(
                    arg < dx::zero, dx::select(exp > dx::zero, result)));
            } else {
                return dx::reinterpret<To>(
                    dx::bit_keep((exp > dx::zero) & (arg > dx::zero), result));
            }
        } else {
            return operator()(cast_t<int32>::operator()(arg));
        }
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<To, A> DPL_VECTORCALL
        fallback(basic_simd<double, A> arg) noexcept
    requires integral<To>
    {
        if constexpr (common_arithmetic_with<int64, To>) {
            using result_type = basic_simd<To, A>;
            constexpr auto d64 = static_cast<int64>(dx::digits_v<double>);
            constexpr auto hidden_bit = dx::one_v<uint64> << d64;
            auto const exp = ilogb(arg);
            auto const umantissa =
                dx::reinterpret<uint64>(arg & dx::mantissa_bits) | hidden_bit;

            auto const shift = dx::abs(exp - d64);
            auto const large = dx::reinterpret<int64>(umantissa << shift);
            auto const small = dx::reinterpret<int64>(umantissa >> shift);
            auto const result = dx::select(exp < d64, small, large);

            if constexpr (signed_integral<To>) {
                return dx::reinterpret<result_type>(dx::negate(
                    arg < dx::zero, dx::select(exp > dx::zero, result)));
            } else {
                return dx::reinterpret<result_type>(
                    dx::bit_keep((exp > dx::zero) & (arg > dx::zero), result));
            }
        } else {
            return operator()(cast_t<int64>::operator()(arg));
        }
    }

    template <basic_simd_element F, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_simd<To, A> fallback(basic_simd<F, A> arg) noexcept {
        return safe_cast(arg);
    }

public:
    template <basic_simd_class From>
    requires simd_type<From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(From arg) noexcept {
        if constexpr (requires { cast<To>(internal::abi<From>, arg); }) {
            if consteval {
                return fallback(arg);
            } else {
                return cast<To>(internal::abi<From>, arg);
            }
        } else {
            return fallback(arg);
        }
    }

    template <simd_type From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(From arg) noexcept {
        if constexpr (requires { cast<To>(internal::abi<From>, arg); }) {
            return cast<To>(internal::abi<From>, arg);
        } else {
            return operator()(dx::to_basic_type(arg));
        }
    }
};

template <typename From, typename To>
concept unqualified_element_castable_to = simd_type<From> && simd_element<To> &&
    requires(From arg) { cast<To>(internal::abi<From>, arg); };

template <simd_element To>
struct cast_t<To> {
public:
    template <simd_type From>
    requires unqualified_element_castable_to<From, To>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(From arg) noexcept
        -> equivalent_simd_as<rebind_simd_t<From, To>> auto {
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
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(From arg) noexcept {
        using result = rebind_simd_t<From, To>;
        using base = cast_t<basic_element_t<To>>;
        return static_cast<result>(base::operator()(dx::to_basic_type(arg)));
    }
};

template <typename From, typename To>
concept unqualified_castable_to =
    simd_type<From> && simd_type<To> && requires(From arg) {
        cast<To>(internal::abi<common_abi_t<From, To>>, arg);
    };

template <basic_simd_type To>
struct cast_t<To> {
public:
    template <basic_simd_type From>
    requires same_abi_simd_as<To, From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        return cast_t<typename To::value_type>::operator()(arg);
    }

    template <simd_type From>
    requires same_abi_simd_as<To, From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        return operator()(dx::to_basic_type(arg));
    }

    template <unqualified_castable_to<To> From>
    requires common_abi_simd_with<To, From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        return cast<To>(internal::abi<common_abi_t<From, To>>, arg);
    }
};

template <simd_type To>
struct cast_t<To> {
public:
    template <unqualified_castable_to<To> From>
    requires common_abi_simd_with<To, From>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
        return cast<To>(internal::abi<common_abi_t<From, To>>, arg);
    }

    template <simd_type From>
    requires (!unqualified_castable_to<From, To>) && requires {
        typename basic_type_t<To>;
        typename invoke_result_t<cast_t<basic_type_t<To>>, From>;
        requires explicitly_convertible_to<
            invoke_result_t<cast_t<basic_type_t<To>>, From>, To>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr To DPL_VECTORCALL operator()(From arg) noexcept {
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
