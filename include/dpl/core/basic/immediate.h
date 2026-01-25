// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/type_traits/remove_const.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <auto V>
struct immediate : integral_constant<remove_const_t<decltype(V)>, V> {
private:
    using base_type DPL_NODEBUG =
        integral_constant<remove_const_t<decltype(V)>, V>;
    using typename base_type::value_type;

public:
    __DPL_HIDE_FROM_ABI constexpr immediate() noexcept = default;

    template <auto U>
    requires core_convertible_to<value_type,
                 typename immediate<U>::value_type> &&
        requires {
            U == V;
            requires U == V;
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator immediate<U>() noexcept {
        return {};
    }

    template <typename T, T U>
    requires core_convertible_to<value_type, T> && (U == V)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator integral_constant<T, U>() noexcept {
        return {};
    }
};

template <auto V>
inline constexpr immediate<V> imm{};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
