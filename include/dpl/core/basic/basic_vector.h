// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/broadcasting.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/initialize.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/immediate/broadcastable_base.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/type_traits/decay.h"
#  include "dpl/std/type_traits/is_same.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename E, typename A>
class basic_vector {
    static_assert(is_same_v<E, decay_t<E>> && is_same_v<A, decay_t<A>>);
    static_assert(simd_element_for<E, A> &&
            is_same_v<simd_element_representation_t<A, E>, E>,
        "Unsupported element type");
    using abi_traits DPL_NODEBUG = simd_abi_traits<E, A>;
    using vector_type DPL_NODEBUG = typename abi_traits::native_vector;

public:
    using value_type = typename abi_traits::element_type;
    using mask_type = basic_mask<E, A>;
    using abi_type = A;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr size_t size() noexcept {
        if constexpr (fixed_width_abi<A>) {
            return abi_traits::size;
        } else {
            return abi_traits::size();
        }
    }

    __DPL_HIDE_FROM_ABI constexpr basic_vector() noexcept
        : basic_vector(datapar::broadcast<basic_vector>(0)) {}

    __DPL_HIDE_FROM_ABI constexpr basic_vector(vector_type vec) noexcept
        : data_(vec) {}

    __DPL_HIDE_FROM_ABI explicit constexpr basic_vector(
        broadcasting_t, E scalar) noexcept
        : basic_vector(datapar::broadcast<E, A>(scalar)) {}

    template <different_from<basic_vector> B>
    requires broadcastable_constant<B, value_type>
    __DPL_HIDE_FROM_ABI constexpr basic_vector(B scalar) noexcept
        : basic_vector(datapar::broadcast<E, A>(static_cast<E>(scalar))) {}

    template <core_convertible_to<E>... Args>
    __DPL_HIDE_FROM_ABI explicit(
        !same_as<internal::cpo_result_t<internal::initialize_t<A>, Args...>,
            basic_vector>) constexpr basic_vector(Args&&... args) noexcept
        : basic_vector(datapar::initialize<basic_vector>(
              __DPL forward<Args>(args)...)) {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr auto operator+(this basic_vector self) noexcept {
        return self.data_;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr auto operator[](
        this basic_vector self, internal::extraction_index auto idx) noexcept {
        return datapar::extract(self, idx);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    explicit constexpr operator vector_type(this basic_vector self) noexcept {
        return self.data_;
    }

private:
    vector_type data_;
};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
