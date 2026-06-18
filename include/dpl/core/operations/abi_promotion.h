// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/std/type_traits/type_identity.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace internal {
#if DPL_COMPILER_GCC
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wnon-template-friend")
#endif

template <fixed_width_abi T>
struct promotion {
    friend consteval auto promote(promotion<T>) noexcept;
};

template <fixed_width_abi T>
struct demotion {
    friend consteval auto demote(demotion<T>) noexcept;
};

template <typename To, typename From>
concept valid_promotion_from = fixed_width_abi<To> && fixed_width_abi<From> &&
    requires { requires To::size > From::size; } && common_abi_with<From, To>;

template <fixed_width_abi From, valid_promotion_from<From> To>
struct define_promotion_t {
    friend consteval auto promote(promotion<From>) noexcept {
        return type_identity<To>{};
    }

    friend consteval auto demote(demotion<To>) noexcept {
        return type_identity<From>{};
    }
};

#if DPL_COMPILER_GCC
DPL_DISABLE_WARNING_POP()
#endif

template <typename To, typename From>
concept define_promotion_from = requires {
    typename internal::define_promotion_t<From, To>;
} && (sizeof(internal::define_promotion_t<From, To>) >= 1);

template <typename T>
using promote_abi DPL_NODEBUG =
    decltype(promote(__DPL datapar::internal::promotion<T>{}));

template <typename T>
using demote_abi DPL_NODEBUG =
    decltype(demote(__DPL datapar::internal::demotion<T>{}));

template <typename T>
concept promotable_abi =
    fixed_width_abi<T> && requires { typename promote_abi<T>; };

template <typename T>
concept demotable_abi =
    fixed_width_abi<T> && requires { typename demote_abi<T>; };

} // namespace internal

DPL_EXPORT template <fixed_width_abi From, internal::define_promotion_from<From> To>
consteval void define_promotion() noexcept {}

#define DPL_DEFINE_ABI_PROMOTION(FROM, TO) \
    static_assert((define_promotion<FROM, TO>(), true))

DPL_EXPORT template <typename>
struct promote_abi {};

DPL_EXPORT template <internal::promotable_abi T>
struct promote_abi<T> : internal::promote_abi<T> {};

DPL_EXPORT template <fixed_width_simd_type T>
requires internal::promotable_abi<typename T::abi_type>
struct promote_abi<T> : promote_abi<typename T::abi_type> {};

DPL_EXPORT template <typename>
struct demote_abi {};

DPL_EXPORT template <internal::demotable_abi T>
struct demote_abi<T> : internal::demote_abi<T> {};

DPL_EXPORT template <fixed_width_simd_type T>
requires internal::demotable_abi<typename T::abi_type>
struct demote_abi<T> : demote_abi<typename T::abi_type> {};

DPL_EXPORT template <typename T>
using promote_abi_t = typename promote_abi<T>::type;

DPL_EXPORT template <typename T>
using demote_abi_t = typename demote_abi<T>::type;

template <size_t N, typename C>
struct concat_target {};

template <size_t N, typename C>
requires (N == C::size)
struct concat_target<N, C> {
    using type DPL_NODEBUG = C;
};

template <size_t N, typename C>
requires (N > C::size) && requires { typename promote_abi_t<C>; }
struct concat_target<N, C> : concat_target<N, promote_abi_t<C>> {};

template <typename... As>
using concat_target_t DPL_NODEBUG =
    typename concat_target<(0zu + ... + As::size),
        promote_abi_t<common_abi_t<As...>>>::type;

template <size_t Target, typename C>
struct split_target {};

template <size_t Target, typename C>
requires (Target == C::size)
struct split_target<Target, C> {
    using type DPL_NODEBUG = C;
};

template <size_t Target, typename C>
requires (Target < C::size) && requires { typename demote_abi_t<C>; }
struct split_target<Target, C> : split_target<Target, demote_abi_t<C>> {};

template <size_t N, typename Source>
using split_target_t DPL_NODEBUG =
    typename split_target<Source::size / N, demote_abi_t<Source>>::type;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
