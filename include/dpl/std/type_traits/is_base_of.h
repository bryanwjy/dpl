// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_base_of)
DPL_EXPORT template <typename Base, typename Derived>
inline constexpr bool is_base_of_v = __is_base_of(Base, Derived);
DPL_EXPORT template <typename Base, typename Derived>
struct is_base_of : bool_constant<__is_base_of(Base, Derived)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_base_of)
DPL_EXPORT template <typename Base, typename Derived>
inline constexpr bool is_base_of_v =
    details::type_traits::unsupported_trait_v<Base, Derived>;
DPL_EXPORT template <typename Base>
inline constexpr bool is_base_of_v<Base, Base> = true;

DPL_EXPORT template <typename Base, typename Derived>
struct is_base_of : bool_constant<is_base_of_v<Base, Derived>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_base_of)

DPL_DEFAULT_NAMESPACE_END
