#pragma once

#include "dpl/config.h"

#include "dpl/concepts.h"

DPL_DEFAULT_NAMESPACE_BEGIN

template <abi_tag T, simd_element U>
requires (T::size >= sizeof(U))
inline constexpr size_t element_count = T::size / sizeof(U);

template <simd_element T>
class __DPL_PUBLIC_TEMPLATE zero_t {
    explicit consteval zero_t() noexcept = default;
};
template <simd_element T>
class __DPL_PUBLIC_TEMPLATE all_t {
    explicit consteval all_t() noexcept = default;
};
template <simd_element T>
class __DPL_PUBLIC_TEMPLATE msb_t {
    explicit consteval msb_t() noexcept = default;
};
template <simd_element T>
class __DPL_PUBLIC_TEMPLATE lsb_t {
    explicit consteval lsb_t() noexcept = default;
};
template <simd_element T>
class __DPL_PUBLIC_TEMPLATE iota_t {
    explicit consteval iota_t() noexcept = default;
};

template <simd_element T>
inline constexpr zero_t<T> zero{};
template <simd_element T>
inline constexpr all_t<T> all{};
template <simd_element T>
inline constexpr msb_t<T> msb{};
template <simd_element T>
inline constexpr lsb_t<T> lsb{};
template <simd_element T>
inline constexpr iota_t<T> iota{};

DPL_DEFAULT_NAMESPACE_END
