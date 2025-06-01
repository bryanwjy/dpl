// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT_BEGIN

struct aligned_t {
    __DPL_HIDE_FROM_ABI explicit constexpr aligned_t() noexcept = default;
};

struct exponent_t {
    __DPL_HIDE_FROM_ABI explicit constexpr exponent_t() noexcept = default;
};

struct mantissa_t {
    __DPL_HIDE_FROM_ABI explicit constexpr mantissa_t() noexcept = default;
};

struct signbit_t {
    __DPL_HIDE_FROM_ABI explicit constexpr signbit_t() noexcept = default;
};

inline constexpr aligned_t aligned{};
inline constexpr exponent_t exponent{};
inline constexpr mantissa_t mantissa{};
inline constexpr signbit_t signbit{};

DPL_EXPORT_END

DPL_DEFAULT_NAMESPACE_END
