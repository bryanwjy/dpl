// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/core/constants/epsilon.h"
#  include "dpl/core/constants/mantissa_bits.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
namespace dx = __DPL datapar;

struct toint_t : broadcastable_base<toint_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr toint_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this toint_t) noexcept {
        return static_cast<T>(1.0) / epsilon_v<T>;
    }
};

inline constexpr toint_t toint{};

struct maxint_t : broadcastable_base<maxint_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr maxint_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this maxint_t) noexcept {
        using uint = dx::unsigned_representation_t<T>;
        return static_cast<T>( __DPL bit_cast<uint>(mantissa_bits_v<T>) + 1);
    }
};

inline constexpr maxint_t maxint{};

struct half_t : broadcastable_base<half_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr half_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this half_t) noexcept {
        return 0.5;
    }
};

inline constexpr half_t half{};

struct underhalf_t : broadcastable_base<underhalf_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr underhalf_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this underhalf_t) noexcept {
        using uint = dx::unsigned_representation_t<T>;
        constexpr auto half_rep = __DPL bit_cast<uint>(static_cast<T>(half));
        return __DPL bit_cast<T>(half_rep - 1);
    }
};

inline constexpr underhalf_t underhalf{};

template <typename T>
struct bit_width_of_t : broadcastable_base<bit_width_of_t<T>> {
    __DPL_HIDE_FROM_ABI explicit constexpr bit_width_of_t() noexcept = default;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator integral auto(this bit_width_of_t) noexcept {
        return sizeof(T) * char_bit_v;
    }
};
template <typename T>
inline constexpr bit_width_of_t<T> bit_width_of{};

struct inv_ln2_t : broadcastable_base<inv_ln2_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr inv_ln2_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this inv_ln2_t) noexcept {
        return 1.442695040888963407359924681001892137426645954152985934135449406931;
    }
};

inline constexpr inv_ln2_t inv_ln2{};

} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
