// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/equality_comparable.h"
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/type_traits/remove_const.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <auto V>
struct immediate : integral_constant<remove_const_t<decltype(V)>, V> {
private:
    using base_type DPL_NODEBUG =
        integral_constant<remove_const_t<decltype(V)>, V>;
    using typename base_type::value_type;

public:
    consteval immediate() noexcept = default;

    template <auto U>
    requires core_convertible_to<value_type,
                 typename immediate<U>::value_type> &&
        requires {
            U == V;
            requires U == V;
        }
    consteval operator immediate<U>(this immediate) noexcept {
        return {};
    }

    template <typename T, T U>
    requires core_convertible_to<value_type, T> && (U == V)
    consteval operator integral_constant<T, U>(this immediate) noexcept {
        return {};
    }

    consteval bool operator==(this immediate, immediate) noexcept = default;

    consteval bool operator<(this immediate, immediate) noexcept
    requires requires { V < V; }
    {
        return false;
    }

    template <auto V2>
    requires requires { V == V2; }
    consteval bool operator==(this immediate<V>, immediate<V2>) noexcept {
        return V == V2;
    }

    template <auto V2>
    requires requires { V < V2; }
    consteval bool operator<(this immediate<V>, immediate<V2>) noexcept {
        return V < V2;
    }
};

DPL_EXPORT template <auto V>
inline constexpr immediate<V> imm{};

DPL_EXPORT template <auto V>
consteval immediate<V> to_immediate(immediate<V> imm) noexcept {
    return imm;
}

namespace internal {
template <typename T>
concept immediate_like = convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> &&
    bool_constant<T() == T::value>::value &&
    bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value;

template <typename T, typename E>
concept immediate_like_of = convertible_to<T, E> && requires {
    typename integral_constant<E, static_cast<E>(T())>;
    requires immediate_like<integral_constant<E, static_cast<E>(T())>>;
};
} // namespace internal

DPL_EXPORT template <internal::immediate_like T>
consteval immediate<T::value> to_immediate(T imm) noexcept {
    return {};
}

DPL_EXPORT template <typename E, internal::immediate_like_of<E> T>
consteval auto to_immediate(T) noexcept -> immediate<static_cast<E>(T())> {
    return {};
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
