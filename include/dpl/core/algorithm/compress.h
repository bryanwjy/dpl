// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/common.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void compress(...) noexcept = delete;

struct DPL_EMPTY_BASES compress_t :
    private algorithm_base<compress_t>,
    private maskable_transform_base<compress_t> {
    using operation_base<compress_t>::operator();
    using maskable_transform_base<compress_t>::operator();
};

template <>
struct operation_signature<compress_t> {
    template <simd_vector T>
    static consteval void operator()(T&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mcompress =
    (!simd_type<S> || same_as<common_abi_t<S, T>, simd_abi_type_t<S>>) &&
    requires(S src, M mask, T val) {
        compress(
            internal::abi<conditional_t<simd_type<S>, S, T>>, src, mask, val);
    };

template <>
struct canonical_impl<compress_t> {
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
        unqualified_canonical_mcompress<S, mask_t<S>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, mask_t<S> mask, T val) noexcept {
        return compress(internal::abi<S>, src, mask, val);
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires canonical_vector<T> &&
        unqualified_canonical_mcompress<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S src, cmask_t<S, M> cmask, T val) noexcept {
        return compress(internal::abi<S>, src, cmask, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcompress<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return compress(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires unqualified_canonical_mcompress<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T val) noexcept {
        return compress(internal::abi<T>, zero, cmask, val);
    }
};

template <typename S, typename M, typename T>
concept unqualified_extended_mcompress = requires {
    {
        compress(internal::declarg<S>(), internal::declarg<M>(),
            internal::declarg<T>())
    } -> equivalent_vector_with<conditional_t<simd_type<S>, S, T>>;
};

template <>
struct extended_impl<compress_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> V>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, V>;

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
        unqualified_extended_mcompress<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return compress( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mcompress<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, cmask_t<S, M> cmask, T&& val) {
        return compress( __DPL forward<S>(src), cmask, __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcompress<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return compress(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mcompress<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return compress(zero, cmask, __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<compress_t> {
public:
    template <simd_vector T>
    static constexpr T operator()(T&& val) noexcept {
        static_assert(!simd_vector<T>,
            "This overload is uninvocable at evaluated contexts");
        return __DPL forward<T>(val);
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires cpo_invocable<exscan_sum_t, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr S DPL_VECTORCALL operator()(
        S src, M mask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        using I = signed_representation_t<simd_element_type_t<M>>;
        auto const simd_size = simd_abi_traits<I, A>::size();
        auto const idx = dx::lane_index<S>();
        auto const rank = fwd::exscan_sum(mask);
        for (auto i = 1zu; i < simd_size; i <<= 1) {
            auto const dist = dx::broadcast<I, A>(static_cast<I>(i));
            auto const perm = idx ^ dist;
            auto const perm_val = dx::permute(val, perm);
            auto const perm_rank = dx::permute(rank, perm);

            auto const move =
                (rank & dist) != dx::zero && perm_rank == (rank ^ dist);
            val = dx::select(move, perm_val, val);
            rank = dx::select(move, perm_rank, rank);
        }

        return dx::select(idx < dx::popcount(mask), val, src);
    }

    template <fixed_width_vector S, fixed_width_mask M, fixed_width_vector T>
    requires cpo_invocable<exscan_sum_t, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S src, M mask, T val) noexcept {
        using A = common_abi_t<M, T>;
        using I = signed_representation_t<simd_element_type_t<M>>;
        auto const rank = fwd::exscan_sum(mask);
        [&val]<size_t J>(this auto self, auto rank, immediate<J>) {
            constexpr auto idx =
                fallback_impl::template butterfly<J>(iota_sequence<S>);
            auto const perm_rank = dx::permute(rank, idx);
            auto const dist = dx::broadcast<I, A>(J);
            auto const move =
                (rank & dist) != dx::zero && perm_rank == (rank ^ dist);

            val = dx::permute(val, move, val, idx);
            if constexpr (J < simd_abi_traits<T>::size) {
                self(dx::select(move, perm_rank, rank), imm<J * 2>);
            }
        }(rank, imm<1zu>);

        return dx::select(
            dx::lane_index<I, A>() < dx::popcount(mask), val, src);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S src, M cmask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        using I = signed_representation_t<simd_element_type_t<T>>;
        constexpr auto rank = []<size_t... Is>(M mask, index_sequence<Is...>) {
            constexpr auto set = static_cast<bitset<M::width>>(mask);
            return index_sequence<fallback_impl::prefix_sum(set, Is)...>{};
        }(cmask, iota_sequence<S>);

        [&val]<size_t J>(this auto self, auto rank, immediate<J>) {
            constexpr auto idx =
                fallback_impl::template butterfly<J>(iota_sequence<S>);
            constexpr auto perm_rank = fallback_impl::permute(rank, idx);
            constexpr auto move =
                fallback_impl::template should_move<J>(perm_rank, rank);

            val = dx::permute(val, move, val, idx);
            if constexpr (J < simd_abi_traits<S>::size) {
                constexpr auto new_idx =
                    fallback_impl::select(move, perm_rank, rank);
                self(new_idx, imm<J * 2>);
            }
        }(rank, imm<1zu>);

        constexpr auto V = (1zu << dx::popcount(cmask)) - 1;
        constexpr make_const_mask_t<S, V> new_mask{};
        return dx::select(new_mask, val, src);
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

    template <size_t... Is, size_t... Js>
    static consteval auto permute(
        index_sequence<Is...> src, index_sequence<Js...> idx) noexcept {
        constexpr size_t array[sizeof...(Is)]{Is...};
        return index_sequence<array[Js]...>{};
    }

    template <size_t N, size_t... Is>
    static consteval auto butterfly(index_sequence<Is...> src) noexcept {
        return index_sequence<(Is ^ N)...>{};
    }

    template <size_t N, size_t... Is, size_t... Js>
    static consteval auto should_move(
        index_sequence<Is...>, index_sequence<Js...>) noexcept {
        constexpr auto width = sizeof...(Is);
        constexpr auto intersection = bitset<width>(((N & Js) != 0)...);
        constexpr auto equality = bitset<width>(((N ^ Js) == Is)...);
        constexpr auto result = intersection & equality;
        if constexpr (integral_bitset_type<bitset<width>>) {
            return const_mask<width, __DPL to_underlying(result)>{};
        } else {
            return const_mask<width, result>{};
        }
    }

    template <size_t... Is, size_t... Js>
    static consteval auto select(
        auto cmask, index_sequence<Is...>, index_sequence<Js...>) noexcept {
        return []<size_t... Ks>(auto cmask, index_sequence<Ks...>) {
            return index_sequence<(cmask[Ks] ? Is : Js)...>{};
        }(cmask, make_index_sequence<sizeof...(Is)>{});
    }
};
} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::compress_t compress{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END