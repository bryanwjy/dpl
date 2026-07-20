// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/type_traits/structured_bindings.h"
#endif

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <auto... Vs>
struct value_pack {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto size() noexcept {
        return sizeof...(Vs);
    }
};

template <auto... Vs>
using constant_type_pack = value_pack<integral_constant<decltype(Vs), Vs>{}...>;

template <auto... Vs>
struct tuple_size<value_pack<Vs...>> : size_constant<sizeof...(Vs)> {};

template <size_t I, auto... Vs>
requires (I < sizeof...(Vs))
struct tuple_element<I, value_pack<Vs...>> {
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && __cpp_pack_indexing >= 202311L
    using type = decltype(Vs...[I]);
#else
private:
    static consteval auto get_type() noexcept {
        return []<auto U, auto... Us, size_t J = 0>(this auto self,
                   value_pack<U, Us...>, size_constant<J> = {}) {
            if constexpr (J == I) {
                return U;
            } else {
                return self(value_pack<Us...>{}, size_constant<J + 1>{});
            }
        }(value_pack<Vs...>{});
    }

public:
    using type = decltype(get_type());
#endif
};

template <size_t I, auto... Vs>
consteval auto get(value_pack<Vs...>) noexcept {
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && __cpp_pack_indexing >= 202311L
    return Vs...[I];
#else
    return []<auto U, auto... Us, size_t J = 0>(
               this auto self, value_pack<U, Us...>, size_constant<J> = {}) {
        if constexpr (J == I) {
            return U;
        } else {
            return self(value_pack<Us...>{}, size_constant<J + 1>{});
        }
    }(value_pack<Vs...>{});
#endif
}

__DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
