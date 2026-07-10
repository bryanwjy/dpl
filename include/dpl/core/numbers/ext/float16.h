// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/ext/common.h" // IWYU pragma: export

#if !DPL_SUPPORTS_FLOAT16

#  if !DPL_MODULES
#    include "dpl/core/details/numbers.h"
#  endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT namespace ext {

#  if DPL_SUPPORTS_STORAGE_FLOAT16
#    define _DPL_FP16_STORAGE_TYPE __fp16
#  else
#    define _DPL_FP16_STORAGE_TYPE uint16
#  endif

class DPL_EMPTY_BASES float16_t :
    public details::numbers::storage16<float16_t, _DPL_FP16_STORAGE_TYPE>,
#  if DPL_SUPPORTS_FLOAT32
    public details::numbers::promotable<float16_t, __DPL float32>,
#  endif
#  if DPL_SUPPORTS_FLOAT64
    public details::numbers::promotable<float16_t, __DPL float64>,
#  endif
#  if DPL_SUPPORTS_FLOAT128
    public details::numbers::promotable<float16_t, __DPL float128>,
#  endif
#  if DPL_SUPPORTS_BFLOAT16
    public details::numbers::promotable<float16_t, __DPL bfloat16>,
#  endif
    public details::numbers::promotable<float16_t, float>,
    public details::numbers::promotable<float16_t, double>,
    public details::numbers::promotable<float16_t, long double>,
    public details::numbers::promotable<char, float16_t>,
    public details::numbers::promotable<signed char, float16_t>,
    public details::numbers::promotable<unsigned char, float16_t>,
    public details::numbers::promotable<short, float16_t>,
    public details::numbers::promotable<unsigned short, float16_t>,
    public details::numbers::promotable<int, float16_t>,
    public details::numbers::promotable<unsigned int, float16_t>,
    public details::numbers::promotable<long, float16_t>,
    public details::numbers::promotable<unsigned long, float16_t>,
    public details::numbers::promotable<long long, float16_t>,
    public details::numbers::promotable<unsigned long long, float16_t>,
    public details::numbers::extended_floating_point_operations<float16_t> {

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) static constexpr storage16
    bits_to_storage(integral auto val) noexcept {
        return storage16( __DPL bit_cast<_DPL_FP16_STORAGE_TYPE>(
            static_cast<uint16>(val & 0xffff)));
    }

private:
    using storage16::value;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) static constexpr storage16 convert(
        float val) noexcept {
#  if DPL_SUPPORTS_STORAGE_FLOAT16
        return storage16{static_cast<__fp16>(val)};
#  else
        constexpr auto msb32 = dx::msb_v<uint32>;
        constexpr auto limitf16 = 0x1p16f;
        constexpr auto inf16 = __DPL to_underlying(
            floating_point_traits<float16_t>::exponent_mask);
        auto const signbit = (__DPL bit_cast<uint32>(val) & msb32) >> 16;
        auto const abs =
            __DPL bit_cast<float>(__DPL bit_cast<uint32>(val) & ~msb32);

        if (abs > limitf16) {
            return bits_to_storage(inf16 | signbit);
        }

        if (auto const isnan = !(abs <= limitf16); isnan) {
            constexpr auto quiet32 = 1u << 22;
            auto const qbit = (__DPL bit_cast<uint32>(val) & quiet32) >> 13;
            return bits_to_storage(inf16 | signbit | qbit);
        }

        constexpr auto shift = 0x1p-112f;
        auto const shifted = __DPL bit_cast<uint32>(abs * shift);
        auto const rounded =
            (((shifted >> 13) & 1u) + (shifted + 0xfffu)) >> 13;
        return bits_to_storage(signbit | rounded);
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) static constexpr storage16 convert(
        double val) noexcept {
        constexpr auto msb64 = dx::msb_v<uint64>;
        constexpr auto limitf16 = 0x1p16;
        constexpr auto inf16 = __DPL to_underlying(
            floating_point_traits<float16_t>::exponent_mask);
        auto const signbit = (__DPL bit_cast<uint64>(val) & msb64) >> 48;
        auto const abs =
            __DPL bit_cast<double>(__DPL bit_cast<uint64>(val) & ~msb64);

        if (abs > limitf16) {
            return bits_to_storage(inf16 | signbit);
        }

        if (auto const isnan = !(abs <= limitf16); isnan) {
            constexpr auto quiet64 = static_cast<uint64>(1ull << 51);
            auto const qbit = (__DPL bit_cast<uint64>(val) & quiet64) >> 42;
            return bits_to_storage(inf16 | signbit | qbit);
        }

        constexpr auto shift = 0x1p-1008;
        constexpr auto round_mask = static_cast<uint64>((0x1ull << 41) - 1ull);
        auto const shifted = __DPL bit_cast<uint64>(abs * shift);
        auto const rounded =
            (((shifted >> 42) & 1) + (shifted + round_mask)) >> 42;
        return bits_to_storage(signbit | rounded);
    }

public:
    __DPL_HIDE_FROM_ABI constexpr float16_t() noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr ~float16_t() noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr float16_t(
        float16_t const&) noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr float16_t& operator=(
        float16_t const&) noexcept = default;

#  define __DPL_F16_ARITHMETIC(OP)                                        \
      DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                    \
      friend constexpr float16_t operator OP(                             \
          float16_t lhs, same_as<float16_t> auto rhs) noexcept {          \
          return static_cast<float16_t>(                                  \
              static_cast<float>(lhs) OP static_cast<float>(rhs));        \
      }                                                                   \
      friend constexpr float16_t& operator OP## =                         \
          (float16_t & lhs, same_as<float16_t> auto rhs) noexcept {       \
          return lhs = static_cast<float16_t>(                            \
                     static_cast<float>(lhs) OP static_cast<float>(rhs)); \
      }                                                                   \
      static_assert(true)

    __DPL_F16_ARITHMETIC(+);
    __DPL_F16_ARITHMETIC(-);
    __DPL_F16_ARITHMETIC(*);
    __DPL_F16_ARITHMETIC(/);

#  undef __DPL_F16_ARITHMETIC

    __DPL_HIDE_FROM_ABI constexpr float16_t(integral auto value) noexcept
        : float16_t(static_cast<float>(value)) {}

    __DPL_HIDE_FROM_ABI constexpr float16_t(float value) noexcept
        : storage16{convert(value)} {}
    __DPL_HIDE_FROM_ABI constexpr float16_t(double value) noexcept
        : storage16{convert(value)} {}
    __DPL_HIDE_FROM_ABI constexpr float16_t(long double value) noexcept
        : float16_t(static_cast<float>(value)) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) constexpr operator float(
        this float16_t self) noexcept {
#  if DPL_SUPPORTS_STORAGE_FLOAT16
        return static_cast<float>(self.value);
#  else
        constexpr auto msb16 = dx::msb_v<uint16>;
        // Not really infinity, it's just an identifying pattern
        constexpr auto inf16_like = 0x0f800000u;
        // align the mantissa
        uint32 const parg =
            static_cast<uint32>(self.value & static_cast<uint16>(~msb16)) << 13;

        if (auto const isfinite = parg < inf16_like; !isfinite) {
            constexpr auto inf32 =
                __DPL bit_cast<uint32>(dx::infinity_v<float>);
            auto const result = inf32 | parg; // deal with NANs
            auto const isneg = msb16 & __DPL bit_cast<uint16>(self.value);
            auto const sign = isneg ? dx::msb_v<uint32> : dx::zero_v<uint32>;
            return __DPL bit_cast<float>(result | sign);
        }

        constexpr auto shift = 0x1p112f;
        // adjusts exp + left shift if subnormal
        auto const shifted = __DPL bit_cast<float>(parg) * shift;
        auto const isneg = msb16 & __DPL bit_cast<uint16>(self.value);
        return isneg ? -shifted : shifted;
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) constexpr operator double(
        this float16_t self) noexcept {
        constexpr auto msb16 = dx::msb_v<uint16>;
        // Not really infinity, it's just an identifying pattern
        constexpr auto inf16_like = 0x1fllu << 52;
        uint16 const bits = __DPL bit_cast<uint16>(self);

        // align the exponent
        uint64 const parg =
            static_cast<uint64>(bits & static_cast<uint16>(~msb16)) << 42;

        if (auto const isfinite = parg < inf16_like; !isfinite) {
            constexpr auto inf64 =
                __DPL bit_cast<uint64>(dx::infinity_v<double>);
            auto const result = inf64 | parg; // deal with NANs
            auto const isneg = msb16 & bits;
            auto const sign = isneg ? dx::msb_v<uint64> : dx::zero_v<uint64>;
            return __DPL bit_cast<double>(result | sign);
        }

        constexpr auto shift = 0x1p1008;
        // adjusts exp + left shift if subnormal
        auto const shifted = __DPL bit_cast<double>(parg) * shift;
        auto const isneg = msb16 & bits;
        return isneg ? -shifted : shifted;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) constexpr operator long double(
        this float16_t self) noexcept {
        return static_cast<long double>(static_cast<double>(self));
    }

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) constexpr operator T(
        this float16_t self) noexcept {
        return static_cast<T>(static_cast<float>(self));
    }

    __DPL_HIDE_FROM_ABI friend constexpr float16_t operator-(
        float16_t self) noexcept {
        auto const val = __DPL bit_cast<uint16>(self);
        return __DPL bit_cast<float16_t>(static_cast<uint16>(val ^ 0x8000));
    }
    __DPL_HIDE_FROM_ABI friend constexpr float16_t operator+(
        float16_t self) noexcept {
        return self;
    }

#  if DPL_SUPPORTS_BFLOAT16
    __DPL_HIDE_FROM_ABI constexpr float16_t(__DPL bfloat16 value) noexcept
        : float16_t(static_cast<float>(value)) {}

    __DPL_HIDE_FROM_ABI constexpr operator __DPL bfloat16(
        this float16_t self) noexcept {
        return static_cast<bfloat16>(static_cast<float>(self));
    }
#  endif
#  if DPL_SUPPORTS_FLOAT32
    __DPL_HIDE_FROM_ABI constexpr float16_t(__DPL float32 value) noexcept
        : float16_t(static_cast<float>(value)) {}

    __DPL_HIDE_FROM_ABI constexpr operator __DPL float32(
        this float16_t self) noexcept {
        return static_cast<__DPL float32>(static_cast<float>(self));
    }
#  endif
#  if DPL_SUPPORTS_FLOAT64
    __DPL_HIDE_FROM_ABI constexpr float16_t(__DPL float64 value) noexcept
        : float16_t(static_cast<double>(value)) {}

    __DPL_HIDE_FROM_ABI constexpr operator __DPL float64(
        this float16_t self) noexcept {
        return static_cast<__DPL float64>(static_cast<double>(self));
    }
#  endif
#  if DPL_SUPPORTS_FLOAT128
    __DPL_HIDE_FROM_ABI constexpr float16_t(__DPL float128 value) noexcept
        : float16_t(static_cast<double>(value)) {}

    __DPL_HIDE_FROM_ABI constexpr operator __DPL float128(
        this float16_t self) noexcept {
        return static_cast<__DPL float128>(static_cast<double>(self));
    }
#  endif

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator==(
        float16_t lhs, same_as<float16_t> auto rhs) noexcept {
#  if DPL_SUPPORTS_STORAGE_FLOAT16
        return self.value == rhs.value;
#  else
        constexpr auto exp16 = 0x7b00;
        auto const labs =
            __DPL bit_cast<uint16>(lhs.value) & dx::value_bits_v<int16>;
        auto const rabs =
            __DPL bit_cast<uint16>(rhs.value) & dx::value_bits_v<int16>;
        return labs <= exp16 && rabs <= exp16 && lhs.value == rhs.value;
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator!=(
        float16_t lhs, same_as<float16_t> auto rhs) noexcept {
#  if DPL_SUPPORTS_STORAGE_FLOAT16
        return lhs.value != rhs.value;
#  else
        constexpr auto exp16 = 0x7b00;
        auto const labs = __DPL bit_cast<uint16>(lhs.value) &
            static_cast<uint16>(~dx::msb_v<uint16>);
        auto const rabs = __DPL bit_cast<uint16>(rhs.value) &
            static_cast<uint16>(~dx::msb_v<uint16>);
        return labs > exp16 || rabs > exp16 || lhs.value != rhs.value;
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator<(
        float16_t lhs, same_as<float16_t> auto rhs) noexcept {
#  if DPL_SUPPORTS_STORAGE_FLOAT16
        return lhs.value < rhs.value;
#  else
        constexpr auto exp16 = 0x7b00;
        constexpr auto man16 = 0x3ff;
        auto const labs = lhs.value & dx::value_bits_v<int16>;
        auto const rabs = rhs.value & dx::value_bits_v<int16>;

        if (labs > exp16 || rabs > exp16) {
            return false;
        }

        auto const lsign = static_cast<int16>(lhs.value & dx::msb_v<int16>);
        auto const rsign = static_cast<int16>(rhs.value & dx::msb_v<int16>);
        auto const lexp = lhs.value & exp16;
        auto const rexp = rhs.value & exp16;
        auto const lman = lhs.value & man16;
        auto const rman = rhs.value & man16;
        return lsign < rsign ||
            (lsign == rsign && (lexp < rexp || (lexp == rexp && lman < rman)));
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator>(
        float16_t lhs, same_as<float16_t> auto rhs) noexcept {
        return rhs < lhs;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator<=(
        float16_t lhs, same_as<float16_t> auto rhs) noexcept {
        return !(rhs < lhs);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator>=(
        float16_t lhs, same_as<float16_t> auto rhs) noexcept {
        return !(lhs < rhs);
    }
};

#  undef _DPL_FP16_STORAGE_TYPE

} // namespace ext

DPL_DEFAULT_NAMESPACE_END
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT namespace ext {
inline namespace literals {
consteval __DPL ext::float16 operator""_f16(long double val) noexcept {
    return static_cast<__DPL ext::float16>(val);
}
} // namespace literals
} // namespace ext

DPL_DEFAULT_NAMESPACE_END
