// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/ext/common.h" // IWYU pragma: export

#if !DPL_SUPPORTS_BFLOAT16

#  include "dpl/core/numbers/details/extended_floating_point_operations.h"
#  include "dpl/core/numbers/details/promotable.h"
#  include "dpl/core/numbers/details/storage16.h"
#  include "dpl/core/numbers/ext/float16.h"
#  include "dpl/core/numbers/floating_point_like.h"

#  if !DPL_MODULES
#    include "dpl/std/concepts/different_from.h"
#    include "dpl/std/concepts/same_as.h"
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace ext {

#  if DPL_SUPPORTS_STORAGE_BFLOAT16
#    define _DPL_BF16_STORAGE_TYPE __bf16
#  else
#    define _DPL_BF16_STORAGE_TYPE uint16
#  endif

class DPL_EMPTY_BASES bfloat16_t :
    public details::numbers::storage16<bfloat16_t, _DPL_BF16_STORAGE_TYPE>,
#  if DPL_SUPPORTS_FLOAT32
    public details::numbers::promotable<bfloat16_t, __DPL float32>,
#  endif
#  if DPL_SUPPORTS_FLOAT64
    public details::numbers::promotable<bfloat16_t, __DPL float64>,
#  endif
#  if DPL_SUPPORTS_FLOAT128
    public details::numbers::promotable<bfloat16_t, __DPL float128>,
#  endif
    public details::numbers::promotable<bfloat16_t, float>,
    public details::numbers::promotable<bfloat16_t, double>,
    public details::numbers::promotable<bfloat16_t, long double>,
    public details::numbers::promotable<char, bfloat16_t>,
    public details::numbers::promotable<signed char, bfloat16_t>,
    public details::numbers::promotable<unsigned char, bfloat16_t>,
    public details::numbers::promotable<short, bfloat16_t>,
    public details::numbers::promotable<unsigned short, bfloat16_t>,
    public details::numbers::promotable<int, bfloat16_t>,
    public details::numbers::promotable<unsigned int, bfloat16_t>,
    public details::numbers::promotable<long, bfloat16_t>,
    public details::numbers::promotable<unsigned long, bfloat16_t>,
    public details::numbers::promotable<long long, bfloat16_t>,
    public details::numbers::promotable<unsigned long long, bfloat16_t>,
    public details::numbers::extended_floating_point_operations<bfloat16_t> {

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) static constexpr storage16
    bits_to_storage(integral auto val) noexcept {
        return storage16( __DPL bit_cast<_DPL_BF16_STORAGE_TYPE>(
            static_cast<uint16>(val & 0xffff)));
    }

private:
    using storage16::value;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) static constexpr storage16 convert(
        float val) noexcept {
#  if DPL_SUPPORTS_STORAGE_BFLOAT16
        return storage16{static_cast<__bf16>(val)};
#  else
        constexpr auto msb32 = dx::msb_v<uint32>;
        constexpr auto inf32 = 0x7f800000u;
        constexpr auto inf16 = 0x7f80;
        auto const signbit = (__DPL bit_cast<uint32>(val) & msb32) >> 16;
        auto const abs =
            __DPL bit_cast<float>(__DPL bit_cast<uint32>(val) & ~msb32);
        if (auto const isnan = !(abs > 0.0f); isnan) {
            constexpr auto quiet32 = 1u << 22;
            auto const qbit = (__DPL bit_cast<uint32>(val) & quiet32) >> 16;
            return bits_to_storage(inf16 | signbit | qbit);
        }

        auto const bits = __DPL bit_cast<uint32>(val);
        auto const lsb = (bits >> 16) & 1;
        auto const bias = 0x7fff + lsb;
        return (bits + bias) >> 16;
#  endif
    }

public:
    __DPL_HIDE_FROM_ABI constexpr bfloat16_t() noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr ~bfloat16_t() noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(
        bfloat16_t const&) noexcept = default;
    __DPL_HIDE_FROM_ABI constexpr bfloat16_t& operator=(
        bfloat16_t const&) noexcept = default;

#  define __DPL_BF16_ARITHMETIC(OP)                                       \
      DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                    \
      friend constexpr bfloat16_t operator OP(                            \
          bfloat16_t lhs, same_as<bfloat16_t> auto rhs) noexcept {        \
          return static_cast<bfloat16_t>(                                 \
              static_cast<float>(lhs) OP static_cast<float>(rhs));        \
      }                                                                   \
      friend constexpr bfloat16_t& operator OP## =                        \
          (bfloat16_t & lhs, same_as<bfloat16_t> auto rhs) noexcept {     \
          return lhs = static_cast<bfloat16_t>(                           \
                     static_cast<float>(lhs) OP static_cast<float>(rhs)); \
      }                                                                   \
      static_assert(true)

    __DPL_BF16_ARITHMETIC(+);
    __DPL_BF16_ARITHMETIC(-);
    __DPL_BF16_ARITHMETIC(*);
    __DPL_BF16_ARITHMETIC(/);

#  undef __DPL_BF16_ARITHMETIC

    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(integral auto value) noexcept
        : bfloat16_t(static_cast<float>(value)) {}

    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(float value) noexcept
        : storage16{convert(value)} {}

    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(double value) noexcept
        : storage16{convert(value)} {}

    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(long double value) noexcept
        : storage16{convert(value)} {}

    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(ext::float16 value) noexcept
        : bfloat16_t(static_cast<float>(value)) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) constexpr operator float(
        this bfloat16_t self) noexcept {
#  if DPL_SUPPORTS_STORAGE_BFLOAT16
        return static_cast<float>(self.value);
#  else
        return __DPL bit_cast<float>(self.value << 16);
#  endif
    }

    template <different_from<float> T>
    requires different_from<T, bfloat16_t> && floating_point_like<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) constexpr operator T(
        this bfloat16_t self) noexcept {
        return static_cast<T>(static_cast<float>(self));
    }

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) constexpr operator T(
        this bfloat16_t self) noexcept {
        return static_cast<T>(static_cast<float>(self));
    }

    __DPL_HIDE_FROM_ABI friend constexpr bfloat16_t operator-(
        bfloat16_t self) noexcept {
        auto const val = __DPL bit_cast<uint16>(self);
        return __DPL bit_cast<bfloat16_t>(static_cast<uint16>(val ^ 0x8000));
    }
    __DPL_HIDE_FROM_ABI friend constexpr bfloat16_t operator+(
        bfloat16_t self) noexcept {
        return self;
    }

#  if DPL_SUPPORTS_FLOAT32
    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(__DPL float32 value) noexcept
        : bfloat16_t(static_cast<float>(value)) {}
#  endif

#  if DPL_SUPPORTS_FLOAT64
    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(__DPL float64 value) noexcept
        : bfloat16_t(static_cast<double>(value)) {}
#  endif

#  if DPL_SUPPORTS_FLOAT128
    __DPL_HIDE_FROM_ABI constexpr bfloat16_t(__DPL float128 value) noexcept
        : bfloat16_t(static_cast<double>(value)) {}
#  endif

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator==(
        bfloat16_t lhs, same_as<bfloat16_t> auto rhs) noexcept {
#  if DPL_SUPPORTS_STORAGE_BFLOAT16
        return lhs.value == rhs.value;
#  else
        return static_cast<float>(lhs) == static_cast<float>(rhs);
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator!=(
        bfloat16_t lhs, same_as<bfloat16_t> auto rhs) noexcept {
#  if DPL_SUPPORTS_STORAGE_BFLOAT16
        return lhs.value != rhs.value;
#  else
        return static_cast<float>(lhs) != static_cast<float>(rhs);
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator<(
        bfloat16_t lhs, same_as<bfloat16_t> auto rhs) noexcept {
#  if DPL_SUPPORTS_STORAGE_BFLOAT16
        return lhs.value < rhs.value;
#  else
        return static_cast<float>(lhs) < static_cast<float>(rhs);
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator>(
        bfloat16_t lhs, same_as<bfloat16_t> auto rhs) noexcept {
        return rhs < lhs;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator<=(
        bfloat16_t lhs, same_as<bfloat16_t> auto rhs) noexcept {
#  if DPL_SUPPORTS_STORAGE_BFLOAT16
        return lhs.value <= rhs.value;
#  else
        return static_cast<float>(lhs) <= static_cast<float>(rhs);
#  endif
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr bool operator>=(
        bfloat16_t lhs, same_as<bfloat16_t> auto rhs) noexcept {
        return rhs <= lhs;
    }
};

#  undef _DPL_BF16_STORAGE_TYPE

} // namespace ext

__DPL_DEFAULT_NAMESPACE_END
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace ext {
inline namespace literals {
consteval __DPL ext::bfloat16 operator""_bf16(long double val) noexcept {
    return static_cast<__DPL ext::bfloat16>(val);
}
} // namespace literals
} // namespace ext

__DPL_DEFAULT_NAMESPACE_END
