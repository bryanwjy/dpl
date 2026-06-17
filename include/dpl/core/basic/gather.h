// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/initialize.h"
#include "dpl/core/basic/internal/abi.h"
#include "dpl/core/basic/internal/iota_sequence.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/immediate/immediate.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void gather(...) noexcept = delete;

struct gather_t :
    private basic_operation_base<gather_t>,
    private maskable_transform_base<gather_t> {
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
    template <typename A, typename E>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<A, E>::size>;
    template <typename A, typename E, imask_t<A, E> M>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<A, E>::size, M>;

public:
    template <simd_abi A, simd_element_for<A> I, simd_element_for<A> E>
    requires integral<I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        E const* ptr, basic_vector<I, A> idx) noexcept {
        if constexpr (unqualified_canonical_gather<A, E const*,
                          basic_vector<I, A>>) {
            return gather(internal::abi<A>, ptr, idx);
        } else {
            // TODO: maybe remove this?
            return []<size_t... Is>(E const* ptr, basic_vector<I, A> idx,
                       index_sequence<Is...>) {
                constexpr auto size = sizeof...(Is);
                auto const zero = E();
                return dx::initialize<E, A>(
                    (Is < idx.size() ? ptr[idx[imm<Is>]] : zero)...);
            }(ptr, idx, iota_sequence<E, A>);
        }
    }

    template <simd_abi A, simd_element_for<A> E, simd_element_for<A> I>
    requires integral<I> &&
        unqualified_canonical_gather<A, basic_vector<E, A>, basic_mask<E, A>,
            E const*, basic_vector<I, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        basic_mask<E, A> mask, E const* ptr, basic_vector<I, A> idx) noexcept {
        return gather(internal::abi<A>, src, mask, ptr, idx);
    }

    template <fixed_width_abi A, simd_element_for<A> E, imask_t<A, E> M,
        simd_element_for<A> I>
    requires integral<I> &&
        unqualified_canonical_gather<A, basic_vector<E, A>, cmask_t<A, E, M>,
            E const*, basic_vector<I, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(basic_vector<E, A> src,
        cmask_t<A, E, M> cmask, E const* ptr, basic_vector<I, A> idx) noexcept {
        return gather(internal::abi<A>, src, cmask, ptr, idx);
    }

    template <simd_abi A, simd_element_for<A> E, simd_element_for<A> I>
    requires integral<I> &&
        unqualified_canonical_gather<A, dx::zero_t, basic_mask<E, A>, E const*,
            basic_vector<I, A>>
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
        static constexpr basic_vector<E, A> operator()(dx::zero_t zero,
            basic_mask<E, A> mask, E const* ptr,
            basic_vector<I, A> idx) noexcept
    requires requires { gather(internal::abi<A>, zero, mask, ptr, idx); }
    {
        return gather(internal::abi<A>, zero, mask, ptr, idx);
    }

    template <fixed_width_abi A, simd_element_for<A> E, imask_t<A, E> M,
        simd_element_for<A> I>
    requires integral<I> &&
        unqualified_canonical_gather<A, dx::zero_t, cmask_t<A, E, M>, E const*,
            basic_vector<I, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(dx::zero_t zero,
        cmask_t<A, E, M> cmask, E const* ptr, basic_vector<I, A> idx) noexcept {
        return gather(internal::abi<A>, zero, cmask, ptr, idx);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::gather_t gather{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
