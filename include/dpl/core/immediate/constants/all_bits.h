// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/is_empty.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#  include "dpl/std/type_traits/sequence.h"
#  include "dpl/std/utility/apply.h"
#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT struct all_bits_t : broadcastable_base<all_bits_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr all_bits_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this all_bits_t) noexcept {
        return static_cast<T>(-1);
    }

    template <typename T>
    requires (!integral<T> && !is_empty_v<T> && is_trivially_copyable_v<T>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this all_bits_t self) noexcept {
        struct bits {
            char data[sizeof(T)];
        };
        return __DPL apply(
            [](auto... args) {
                return __DPL bit_cast<T>(
                    bits{static_cast<char>((__DPL ignore = args, -1))...});
            },
            make_index_sequence<sizeof(T)>{});
    }
};

DPL_EXPORT inline constexpr all_bits_t all_bits{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<all_bits_t, T>
inline constexpr auto all_bits_v = static_cast<T>(all_bits);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
