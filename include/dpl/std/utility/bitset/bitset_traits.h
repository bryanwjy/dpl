// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/equality_comparable.h"
#  include "dpl/std/type_traits/structured_bindings.h"
#  include "dpl/std/type_traits/unsigned_integral_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t W>
class bitset;

template <size_t W>
struct tuple_size<bitset<W>> : size_constant<W> {};

template <size_t I, size_t W>
struct tuple_element<I, bitset<W>> {
    using type DPL_NODEBUG = bool;
};

template <size_t I, size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto get(
    bitset<W> const& value) noexcept {
    return value[I];
}

struct low_bits_t {
    explicit constexpr low_bits_t() noexcept = default;
};

struct high_bits_t {
    explicit constexpr high_bits_t() noexcept = default;
};

struct bit_range_t {
    explicit constexpr bit_range_t() noexcept = default;
};

inline constexpr low_bits_t low_bits{};
inline constexpr high_bits_t high_bits{};
inline constexpr bit_range_t bit_range{};

namespace details::utility {
template <typename T>
inline constexpr bool is_bitset_v = false;
template <typename T>
inline constexpr bool is_bitset_v<T const> = is_bitset_v<T>;
template <typename T>
inline constexpr bool is_bitset_v<T volatile> = is_bitset_v<T>;
template <typename T>
inline constexpr bool is_bitset_v<T const volatile> = is_bitset_v<T>;
template <size_t W>
inline constexpr bool is_bitset_v<bitset<W>> = true;

template <typename T>
concept bitset_type = is_bitset_v<T>;

template <typename T>
concept bitset_constant_like = requires { T::value; } &&
    bitset_type<decltype(T::value)> && convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> && (T() == T::value) &&
    (static_cast<decltype(T::value)>(T()) == T::value);

consteval size_t ceil_width(size_t val) noexcept {
    if (val < __DPL char_bit_v) {
        return __DPL char_bit_v;
    }

    return 1zu << (__DPL bit_width(val) - __DPL has_single_bit(val));
}

template <size_t W>
struct bitset_storage {
    using underlying_type = size_t[W / __DPL type_bit_v<size_t> +
        (W % (__DPL type_bit_v<size_t>) != 0)];
    underlying_type storage_;
};

template <size_t W>
requires requires { typename unsigned_integral_type_t<utility::ceil_width(W)>; }
struct bitset_storage<W> {
    using underlying_type = unsigned_integral_type_t<utility::ceil_width(W)>;
    underlying_type value_;
};

} // namespace details::utility

__DPL_DEFAULT_NAMESPACE_END
