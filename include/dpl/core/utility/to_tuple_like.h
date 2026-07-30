// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/tuple_access.h"
#  include "dpl/core/concepts/simd_tuple.h"
#  include "dpl/std/concepts/tuple_like.h"
#  include "dpl/std/type_traits/is_reference.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace internal {
template <typename T>
class tuple_reference {
    static_assert(is_reference_v<T>);
    static_assert(simd_tuple<T>);

public:
    __DPL_HIDE_FROM_ABI constexpr tuple_reference(T ref) noexcept
        : reference(ref) {}

    template <size_t I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    friend constexpr decltype(auto) get(tuple_reference instance) noexcept {
        return dx::get_element<I>(static_cast<T>(instance.reference));
    }

private:
    T reference;
};

} // namespace internal

template <simd_tuple T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr decltype(auto) to_tuple_like(
    T&& tuple DPL_LIFETIMEBOUND) noexcept {
    if constexpr (tuple_like<T>) {
        return __DPL forward<T>(tuple);
    } else {
        // MSVC has weird behaviour when using () instead of {}
        return internal::tuple_reference<T&&>{__DPL forward<T>(tuple)};
    }
}
} // namespace datapar

template <typename T>
struct tuple_size<datapar::internal::tuple_reference<T>> :
    tuple_size<remove_cvref_t<T>> {};

template <size_t I, typename T>
struct tuple_element<I, datapar::internal::tuple_reference<T>> :
    tuple_element<I, remove_cvref_t<T>> {};

__DPL_DEFAULT_NAMESPACE_END
