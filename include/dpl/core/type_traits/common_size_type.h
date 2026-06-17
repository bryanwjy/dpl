// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/common_with.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/common_type.h"
#  include "dpl/std/type_traits/make_signed.h"
#  include "dpl/std/type_traits/make_unsigned.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename... T>
struct common_size_type {};
DPL_EXPORT template <typename... Ts>
using common_size_type_t = typename common_size_type<Ts...>::type;

DPL_EXPORT template <typename T>
struct common_size_type<T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <typename T>
struct common_size_type<T, T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <typename A, typename B>
requires (sizeof(A) == sizeof(B))
struct common_size_type<A, B> {
private:
    static consteval auto choose_type() noexcept {
        if constexpr (signed_integral<A> && signed_integral<B>) {
            return make_signed_t<bit_type_t<sizeof(A) * char_bit_v>>{};
        } else if constexpr (unsigned_integral<A> && unsigned_integral<B>) {
            return make_unsigned_t<bit_type_t<sizeof(A) * char_bit_v>>{};
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
                return bit_type<sizeof(A) * char_bit_v>{};
            }
        } else {
            return bit_type<sizeof(A) * char_bit_v>{};
        }
    }

public:
    using type DPL_NODEBUG = typename decltype(choose_type())::type;
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
