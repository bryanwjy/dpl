// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/broadcastable_base.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/initialize.h"
#include "dpl/core/basic/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/common_order_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/type_traits/element_count.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/is_enum.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <simd_element E, simd_abi A>
class basic_simd<E, A> {
    using vector_type DPL_NODEBUG = typename A::template native_type<E>;

public:
    using value_type = E;
    using abi_type = A;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept { return element_count<E, A>; }

    __DPL_HIDE_FROM_ABI constexpr basic_simd() noexcept
        : basic_simd(datapar::broadcast<basic_simd>(0)) {}

    __DPL_HIDE_FROM_ABI constexpr basic_simd(vector_type vec) noexcept
        : data_(vec) {}

    template <different_from<E> E2>
    requires simd_element<E2> && common_order_with<E2, E>
    __DPL_HIDE_FROM_ABI constexpr basic_simd(
        basic_simd<E2, abi_type> other) noexcept
        : basic_simd(datapar::reinterpret<basic_simd>(other)) {}

    template <different_from<basic_simd> B>
    requires broadcastable_to<B, A> && broadcastable_constant<B, value_type>
    __DPL_HIDE_FROM_ABI constexpr basic_simd(B scalar) noexcept
        : basic_simd(datapar::broadcast<A>(scalar)) {}

    template <typename... Args>
    requires regular_invocable<internal::initialize_t<A>, Args...>
    __DPL_HIDE_FROM_ABI constexpr basic_simd(Args&&... args) noexcept
        : basic_simd(datapar::initialize<A>(__DPL forward<Args>(args)...)) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr auto DPL_VECTORCALL operator+(this basic_simd self) noexcept {
        return self.data_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr auto operator[](
        this basic_simd self, internal::extraction_index auto idx) noexcept {
        assert(idx < element_count<basic_simd>);
        return datapar::extract(self, idx);
    }

private:
    vector_type data_;
};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
