// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/broadcastable_base.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/initialize.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/common_order_with.h" // IWYU pragma: keep
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element_for.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/different_from.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename E, simd_abi A>
requires simd_element_for<E, A>
class basic_simd<E, A> {
    using traits DPL_NODEBUG = simd_abi_traits<E, A>;
    using vector_type DPL_NODEBUG = typename traits::native_type;

public:
    using value_type = typename traits::element_type;
    using abi_type = A;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept {
        if constexpr (fixed_width_abi<A>) {
            return traits::size;
        } else {
            return traits::size();
        }
    }

    __DPL_HIDE_FROM_ABI constexpr basic_simd() noexcept
        : basic_simd(datapar::broadcast<basic_simd>(0)) {}

    __DPL_HIDE_FROM_ABI constexpr basic_simd(vector_type vec) noexcept
        : data_(vec) {}

    template <different_from<basic_simd> B>
    requires broadcastable_constant<B, value_type>
    __DPL_HIDE_FROM_ABI constexpr basic_simd(B scalar) noexcept
        : basic_simd(datapar::broadcast<E, A>(static_cast<E>(scalar))) {}

    template <core_convertible_to<E>... Args>
    __DPL_HIDE_FROM_ABI explicit(
        !same_as<invoke_result_t<internal::initialize_t<A>, Args...>,
            basic_simd>) constexpr basic_simd(Args&&... args) noexcept
        : basic_simd(
              datapar::initialize<basic_simd>(__DPL forward<Args>(args)...)) {
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr auto operator+(this basic_simd self) noexcept {
        return self.data_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr auto operator[](
        this basic_simd self, internal::extraction_index auto idx) noexcept {
        return datapar::extract(self, idx);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit constexpr operator vector_type(this basic_simd self) noexcept {

        return self.data_;
    }

private:
    vector_type data_;
};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
