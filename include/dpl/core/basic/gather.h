// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/internal/abi.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/immediate/constants/zero.h"

#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void gather(...) noexcept = delete;

struct gather_t :
    public basic_operation_base<gather_t>,
    public maskable_transform_base<gather_t> {
    using operation_base<gather_t>::operator();
    using maskable_transform_base<gather_t>::operator();
};

template <>
struct operation_signature<gather_t> {
    template <canonical_vector T>
    requires integral<simd_element_type_t<T>>
    static consteval void operator()(
        simd_element_for<simd_abi_type_t<T>> auto const*, T) noexcept {}
};

template <typename A, typename... Ts>
concept unqualified_canonical_gather = requires {
    {
        gather(internal::abi<A>, internal::declarg<Ts>()...)
    } -> canonical_vector;
};

template <>
struct canonical_impl<gather_t> {

private:
    template <typename E, typename A>
    using mask_t DPL_NODEBUG = make_canonical_mask_t<E, A>;

public:
    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E>
    requires same_as<int, simd_element_type_t<I>> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(
        E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E>
    requires same_as<int, simd_element_type_t<I>> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, rebind_simd_t<I, E>,
            mask_t<E, simd_abi_type_t<I>>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(rebind_simd_t<I, E> src,
        mask_t<E, simd_abi_type_t<I>> mask, E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, src, mask, ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E,
        const_mask_for<rebind_simd_t<I, E>> M>
    requires same_as<int, simd_element_type_t<I>> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, rebind_simd_t<I, E>,
            launder_cmask_t<rebind_simd_t<I, E>, M>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(
        rebind_simd_t<I, E> src, M cmask, E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, src,
            dx::to_const_mask<rebind_simd_t<I, E>>(cmask), ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E>
    requires same_as<int, simd_element_type_t<I>> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, dx::zero_t,
            mask_t<E, simd_abi_type_t<I>>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(dx::zero_t zero,
        mask_t<E, simd_abi_type_t<I>> mask, E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, zero, mask, ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E,
        const_mask_for<rebind_simd_t<I, E>> M>
    requires unqualified_canonical_gather<simd_abi_type_t<I>, dx::zero_t,
        launder_cmask_t<rebind_simd_t<I, E>, M>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(
        dx::zero_t zero, M cmask, E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, zero,
            dx::to_const_mask<rebind_simd_t<I, E>>(cmask), ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E>
    requires same_as<ptrdiff_t, simd_element_type_t<I>> &&
        different_from<ptrdiff_t, int> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(
        E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E>
    requires same_as<ptrdiff_t, simd_element_type_t<I>> &&
        different_from<ptrdiff_t, int> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, rebind_simd_t<I, E>,
            mask_t<E, simd_abi_type_t<I>>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(rebind_simd_t<I, E> src,
        mask_t<E, simd_abi_type_t<I>> mask, E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, src, mask, ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E,
        const_mask_for<rebind_simd_t<I, E>> M>
    requires same_as<ptrdiff_t, simd_element_type_t<I>> &&
        different_from<ptrdiff_t, int> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, rebind_simd_t<I, E>,
            launder_cmask_t<rebind_simd_t<I, E>, M>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(
        rebind_simd_t<I, E> src, M cmask, E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, src,
            dx::to_const_mask<rebind_simd_t<I, E>>(cmask), ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E>
    requires same_as<ptrdiff_t, simd_element_type_t<I>> &&
        different_from<ptrdiff_t, int> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, dx::zero_t,
            mask_t<E, simd_abi_type_t<I>>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(dx::zero_t zero,
        mask_t<E, simd_abi_type_t<I>> mask, E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, zero, mask, ptr, idx);
    }

    template <canonical_vector I, simd_element_for<simd_abi_type_t<I>> E,
        const_mask_for<rebind_simd_t<I, E>> M>
    requires same_as<ptrdiff_t, simd_element_type_t<I>> &&
        different_from<ptrdiff_t, int> &&
        unqualified_canonical_gather<simd_abi_type_t<I>, dx::zero_t,
            launder_cmask_t<rebind_simd_t<I, E>, M>, E const*, I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<I, E> operator()(
        dx::zero_t zero, M cmask, E const* ptr, I idx) noexcept {
        return gather(internal::abi<I>, zero,
            dx::to_const_mask<rebind_simd_t<I, E>>(cmask), ptr, idx);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::gather_t gather{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
