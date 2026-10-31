// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_base_of.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace internal {
struct accuracy_base_t {
    __DPL_HIDE_FROM_ABI constexpr ~accuracy_base_t() = default;
};

template <typename T>
concept accuracy_tag =
    is_base_of_v<accuracy_base_t, T> && !same_as<accuracy_base_t, T>;
} // namespace internal

namespace accuracy {
struct precise_t : public internal::accuracy_base_t {
    __DPL_HIDE_FROM_ABI explicit constexpr precise_t() noexcept = default;

    consteval bool operator==(
        this precise_t, internal::accuracy_tag auto) noexcept {
        return false;
    }
    consteval bool operator!=(
        this precise_t, internal::accuracy_tag auto) noexcept {
        return true;
    }
    consteval bool operator==(this precise_t, precise_t) noexcept {
        return true;
    }
    consteval bool operator!=(this precise_t, precise_t) noexcept {
        return false;
    }
};

struct approximate_t : public internal::accuracy_base_t {
    __DPL_HIDE_FROM_ABI explicit constexpr approximate_t() noexcept = default;
    consteval bool operator==(
        this approximate_t, internal::accuracy_tag auto) noexcept {
        return false;
    }
    consteval bool operator!=(
        this approximate_t, internal::accuracy_tag auto) noexcept {
        return true;
    }
    consteval bool operator==(this approximate_t, approximate_t) noexcept {
        return true;
    }
    consteval bool operator!=(this approximate_t, approximate_t) noexcept {
        return false;
    }
};

inline constexpr approximate_t approximate{};
inline constexpr precise_t precise{};
} // namespace accuracy
} // namespace datapar

namespace accuracy = datapar::accuracy; // NOLINT
__DPL_DEFAULT_NAMESPACE_END
