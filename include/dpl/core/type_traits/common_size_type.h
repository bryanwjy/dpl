// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/common_with.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/common_type.h"
#  include "dpl/std/type_traits/make_signed.h"
#  include "dpl/std/type_traits/make_unsigned.h"
#  include "dpl/std/type_traits/signed_integral_type.h"
#  include "dpl/std/type_traits/underlying_type.h"
#  include "dpl/std/type_traits/unsigned_integral_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename... T>
struct common_size_type {};
template <typename... Ts>
using common_size_type_t = typename common_size_type<Ts...>::type;

template <typename T>
struct common_size_type<T> {
    using type DPL_NODEBUG = T;
};

template <typename T>
struct common_size_type<T, T> {
    using type DPL_NODEBUG = T;
};

template <typename A, typename B>
requires (sizeof(A) == sizeof(B))
struct common_size_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        constexpr auto width = __DPL type_bit_v<A>;
        if constexpr (signed_integral<A> && signed_integral<B>) {
            return signed_integral_type<width>{};
        } else if constexpr (unsigned_integral<A> && unsigned_integral<B>) {
            return unsigned_integral_type<width>{};
        } else if constexpr (enumeration<A> && enumeration<B>) {
            return common_size_type<underlying_type_t<A>,
                underlying_type_t<B>>{};
        } else if constexpr (enumeration<A>) {
            return common_size_type<underlying_type_t<A>, B>{};
        } else if constexpr (enumeration<B>) {
            return common_size_type<A, underlying_type_t<B>>{};
        } else if constexpr (common_with<A, B>) {
            if constexpr (sizeof(common_type_t<A, B>) == sizeof(A)) {
                return common_type<A, B>{};
            } else {
                return unsigned_integral_type<width>{};
            }
        } else {
            return unsigned_integral_type<width>{};
        }
    }

public:
    using type DPL_NODEBUG = typename decltype(choose_type())::type;
};
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
