// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#if !DPL_MODULES
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/derived_from.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT
struct broadcastable_base {
protected:
    __DPL_HIDE_FROM_ABI constexpr ~broadcastable_base() noexcept = default;

public:
    template <typename S, simd_class T>
    requires convertible_to<S, typename T::value_type>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this S self) noexcept {
        return datapar::broadcast<T>(static_cast<typename T::value_type>(self));
    }
};

DPL_EXPORT template <typename T, typename ValueType>
concept broadcastable_constant = derived_from<T, broadcastable_base> &&
    is_empty_v<T> && semiregular<T> && convertible_to<T, ValueType> &&
    requires { typename immediate<static_cast<ValueType>(T{})>; };

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
