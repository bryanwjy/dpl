// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/common.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void expand(...) noexcept = delete;

struct DPL_EMPTY_BASES expand_t :
    private algorithm_base<expand_t>,
    private maskable_transform_base<expand_t> {
    using operation_base<expand_t>::operator();
    using maskable_transform_base<expand_t>::operator();
};

template <>
struct operation_signature<expand_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mexpand =
    (!simd_type<S> || same_as<common_abi_t<S, T>, simd_abi_type_t<S>>) &&
    requires(S src, M mask, T val) {
        expand(
            internal::abi<conditional_t<simd_type<S>, S, T>>, src, mask, val);
    };

template <>
struct canonical_impl<expand_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

public:
    template <simd_vector T>
    static constexpr T operator()(T&& val) noexcept {
        static_assert(!simd_vector<T>,
            "This overload is uninvocable at evaluated contexts");
        return __DPL forward<T>(val);
    }

    template <canonical_vector S, common_vector_with<S> T>
    requires canonical_vector<T> &&
        unqualified_canonical_mexpand<S, mask_t<S>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, mask_t<S> mask, T val) noexcept {
        return expand(internal::abi<S>, src, mask, val);
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires canonical_vector<T> &&
        unqualified_canonical_mexpand<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, cmask_t<S, M> cmask, T val) noexcept {
        return expand(internal::abi<S>, src, cmask, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mexpand<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return expand(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires unqualified_canonical_mexpand<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T val) noexcept {
        return expand(internal::abi<T>, zero, cmask, val);
    }
};

template <typename S, typename M, typename T>
concept unqualified_extended_mexpand = requires {
    {
        expand(internal::declarg<S>(), internal::declarg<M>(),
            internal::declarg<T>())
    } -> equivalent_vector_with<conditional_t<simd_type<S>, S, T>>;
};

template <>
struct extended_impl<expand_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_type_t<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_type_t<T>::size, V>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_vector T>
    static constexpr T operator()(T&& val) noexcept {
        static_assert(!simd_vector<T>,
            "This overload is uninvocable at evaluated contexts");
        return __DPL forward<T>(val);
    }

    template <simd_vector S, common_vector_with<S> T,
        equivalent_mask_with<mask_t<S>> M>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mexpand<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return expand( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mexpand<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, cmask_t<S, M> cmask, T&& val) {
        return expand( __DPL forward<S>(src), cmask, __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mexpand<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return expand(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mexpand<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return expand(zero, cmask, __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<expand_t> {

private:
    template <size_t W>
    static consteval size_t prefix_sum(
        bitset<W> const& set, size_t idx) noexcept {
        auto result = 0zu;
        for (auto i = 0zu; i < idx; ++i) {
            result += set.test(i);
        }
        return result;
    }

    template <typename M>
    static consteval auto prefix_sum(M cmask) noexcept {
        return []<size_t... Is>(M mask, index_sequence<Is...>) {
            constexpr auto set = static_cast<bitset<M::width>>(mask);
            return index_sequence<fallback_impl::prefix_sum(set, Is)...>{};
        }(cmask, make_index_sequence<M::width>{});
    }

public:
    template <simd_vector T>
    static constexpr T operator()(T&& val) noexcept {
        static_assert(!simd_vector<T>,
            "This overload is uninvocable at evaluated contexts");
        return __DPL forward<T>(val);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires cpo_invocable<exscan_sum_t, M> &&
        cpo_invocable<permute_t, S, M, T, cpo_result_t<exscan_sum_t, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr S DPL_VECTORCALL operator()(
        S&& src, M&& mask, T&& val) noexcept {
        return dx::permute(__DPL forward<S>(src), __DPL forward<S>(mask),
            __DPL forward<T>(val), fwd::exscan_sum(mask));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires cpo_invocable<permute_t, S, M, T,
        decltype(fallback_impl::prefix_sum(internal::declarg<M>()))>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S src, M cmask, T val) noexcept {
        constexpr auto seq = fallback_impl::prefix_sum(cmask);
        return dx::permute(src, cmask, val, seq);
    }

    template <simd_mask M, simd_vector R>
    requires requires {
        operator()(internal::declarg<canonical_type_t<R>>(),
            internal::declarg<M>(), internal::declarg<R>());
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask,
        R&& val) noexcept(canonical_mask<M> && canonical_vector<R>) {
        return operator()(dx::broadcast<R>(dx::zero), __DPL forward<M>(mask),
            __DPL forward<R>(val));
    }

    template <simd_vector R, const_mask_for<R> M>
    requires requires {
        operator()(internal::declarg<canonical_type_t<R>>(),
            internal::declarg<M>(), internal::declarg<R>());
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, R&& val) noexcept {
        return operator()(
            dx::broadcast<R>(dx::zero), cmask, __DPL forward<R>(val));
    }
};
} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::expand_t expand{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
