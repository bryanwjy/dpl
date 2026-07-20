// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/type_traits/structured_bindings.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename... Ts>
struct type_pack {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto size() noexcept {
        return sizeof...(Ts);
    }
};

template <typename... Ts>
struct tuple_size<type_pack<Ts...>> : size_constant<sizeof...(Ts)> {};

template <size_t I, typename... Ts>
requires (I < sizeof...(Ts))
struct tuple_element<I, type_pack<Ts...>> {
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && __cpp_pack_indexing >= 202311L
    using type = type_identity<Ts...[I]>;
#else
private:
    static consteval auto get_type() noexcept {
        return []<typename U, typename... Us, size_t J = 0>(
                   this auto self, type_pack<U, Us...>, size_constant<J> = {}) {
            if constexpr (J == I) {
                return type_identity<U>{};
            } else {
                return self(type_pack<Us...>{}, size_constant<J + 1>{});
            }
        }(type_pack<Ts...>{});
    }

public:
    using type = decltype(get_type());
#endif
};

template <size_t I, typename... Ts>
consteval auto get(type_pack<Ts...>) noexcept {
    return tuple_element_t<I, type_pack<Ts...>>{};
}

__DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
