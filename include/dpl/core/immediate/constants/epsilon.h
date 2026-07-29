// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/one.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/floating_point_like.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

inline constexpr struct epsilon_t : broadcastable_base<epsilon_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr epsilon_t() noexcept = default;

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this epsilon_t) noexcept {
        using bitset_t = bitset<__DPL type_bit_v<T>>;
        constexpr auto one = bitset_t(1);
        constexpr auto next = __DPL bit_cast<bitset_t>(one_v<T>) | one;
        return __DPL bit_cast<T>(next) - one_v<T>;
    }
} epsilon{};

template <typename T>
requires explicitly_convertible_to<epsilon_t, T>
inline constexpr auto epsilon_v = static_cast<T>(epsilon);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
