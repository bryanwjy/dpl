// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/bitset/concepts.h"
#include "dpl/std/utility/bitset/countl.h"
#include "dpl/std/utility/bitset/countr.h"
#include "dpl/std/utility/bitset/integral_bitset.h"
#include "dpl/std/utility/bitset/large_bitset.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/type_traits/has_unique_object_representations.h"
#  include "dpl/std/type_traits/is_scalar.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace details::utility {

template <floating_point T>
requires (sizeof(T) == sizeof(bitset<80>))
struct float80 {
    static_assert(80 % __DPL char_bit_v == 0);
    static constexpr auto bytes = 80zu / __DPL char_bit_v;
    struct val_t {
        unsigned char data[bytes];
    } val;
    struct padding_t {
        unsigned char data[sizeof(T) - bytes];
    } padding;
};

template <typename T>
consteval bool is_x87_float80() noexcept {
    if constexpr (floating_point<T> && requires { typename float80<T>; }) {
        constexpr auto reinterpretx87 = [](T val) {
            float80<T> copy{}; // clear the upper padding bits
            copy.val = __DPL bit_cast<float80<T>>(val).val;
            return __DPL bit_cast<bitset<80>>(copy);
        };
        constexpr auto one = static_cast<T>(1.0);
        constexpr auto nzero = static_cast<T>(-0.0);
        return __DPL countr_zero(reinterpretx87(one)) == 63 &&
            __DPL countl_one(reinterpretx87(nzero)) == 1;
    } else {
        return false;
    }
}
} // namespace details::utility

template <typename T>
requires is_trivially_copyable_v<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto to_bit_representation(
    T const& val) noexcept {
    if constexpr (details::utility::is_x87_float80<T>()) {
        using details::utility::float80;
        float80<T> copy{}; // clear the upper padding bits
        copy.val = __DPL bit_cast<float80<T>>(val).val;
        return __DPL bit_cast<bitset<80>>(copy);
    } else {
        return __DPL bit_cast<bitset<__DPL type_bit_v<T>>>(val);
    }
}

template <typename T>
struct bit_representation {};

template <typename T>
using bit_representation_t = typename bit_representation<T>::type;

template <typename T>
requires requires(T const& val) {
    { __DPL to_bit_representation(val) } noexcept;
}
struct bit_representation<T> {
    using type DPL_NODEBUG =
        decltype(__DPL to_bit_representation(*(T const*)0));
};

__DPL_DEFAULT_NAMESPACE_END
