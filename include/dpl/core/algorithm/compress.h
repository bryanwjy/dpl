// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/scan/exscan_sum.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void compress(...) noexcept = delete;

struct compress_t : public algorithm_base<compress_t> {
    using operation_base<compress_t>::operator();
};

template <>
struct operation_signature<compress_t> {
    template <simd_vector T, const_mask_for<T> M, equivalent_vector_with<T> S>
    static consteval void operator()(T&&, M, S&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M, equivalent_vector_with<T> S>
    static consteval void operator()(T&&, M, S&&) noexcept {}

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> S>
    static consteval void operator()(T&&, M, S&&) noexcept {}

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> S>
    static consteval void operator()(T&&, M, S&&) noexcept {}
};

template <typename T, typename M, typename S>
concept unqualified_canonical_compress = requires(T val, M mask, S src) {
    { compress(internal::abi<T>, val, mask, src) } -> same_as<T>;
};
template <typename T, typename M>
concept unqualified_canonical_zcompress = requires(T val, M mask) {
    { compress(internal::abi<T>, val, mask, dx::zero) } -> same_as<T>;
};

template <>
struct canonical_impl<compress_t> {
    template <canonical_vector T, same_as<T> S>
    requires unqualified_canonical_compress<T, simd_mask_type_t<T>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, simd_mask_type_t<T> mask, S src) noexcept {
        return compress(internal::abi<T>, val, mask, src);
    }

    template <canonical_vector T, const_mask_for<T> M, same_as<T> S>
    requires unqualified_canonical_compress<T, launder_cmask_t<T, M>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, M mask, S src) noexcept {
        return compress(internal::abi<T>, val, dx::to_const_mask<T>(mask), src);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zcompress<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        T val, simd_mask_type_t<T> mask, dx::zero_t zero) noexcept {
        return compress(internal::abi<T>, val, mask, zero);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zcompress<T, launder_cmask_t<T, M>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, M mask, dx::zero_t zero) noexcept {
        return compress(
            internal::abi<T>, val, dx::to_const_mask<T>(mask), zero);
    }
};

template <typename T, typename M, typename S>
concept unqualified_extended_compress = requires {
    {
        compress(internal::declarg<T>(), internal::declarg<M>(),
            internal::declarg<S>())
    } -> equivalent_vector_with<S>;
};

template <typename T, typename M>
concept unqualified_extended_zcompress = requires {
    {
        compress(internal::declarg<T>(), internal::declarg<M>(), dx::zero)
    } -> equivalent_vector_with<T>;
};

template <>
struct extended_impl<compress_t> {
    template <simd_vector T, exact_mask_for<T> M, equivalent_vector_with<T> S>
    requires (extended_vector<T> || extended_mask<T> || extended_vector<S>) &&
        unqualified_extended_compress<T, M, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask, S&& src) {
        return compress(__DPL forward<T>(val), __DPL forward<M>(mask),
            __DPL forward<S>(src));
    }

    template <simd_vector T, const_mask_for<T> M, equivalent_vector_with<T> S>
    requires (extended_vector<T> || extended_vector<S>) &&
        unqualified_extended_compress<T, launder_cmask_t<T, M>, S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask, S&& src) {
        return compress(__DPL forward<T>(val), dx::to_const_mask<T>(mask),
            __DPL forward<S>(src));
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_vector<T> || extended_mask<T>) &&
        unqualified_extended_zcompress<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M&& mask, dx::zero_t zero) {
        return compress(__DPL forward<T>(val), __DPL forward<M>(mask), zero);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zcompress<T, M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, M mask, dx::zero_t zero) {
        return compress(
            __DPL forward<T>(val), dx::to_const_mask<T>(mask), zero);
    }
};

template <>
struct fallback_impl<compress_t> {

    // These are too complicated to allow extended fallbacks

    template <canonical_vector T>
    requires (!fixed_width_abi<simd_abi_type_t<T>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(
        T val, simd_mask_type_t<T> mask, type_identity_t<T> src) noexcept {
        using A = simd_abi_type_t<T>;
        using vidx_t = signed_canonical_vector_t<T>;
        auto const simd_size = simd_abi_traits<vidx_t>::size();
        auto const iota = dx::lane_index<vidx_t>();
        auto const vone = dx::broadcast<vidx_t>(dx::one);
        auto const pop = dx::popcount(mask);
        auto const active =
            dx::exscan_sum(dx::select(mask, vone, dx::zero), dx::zero);
        auto const inactive =
            dx::exscan_sum(dx::select(mask, dx::zero, vone), pop);
        auto const max = dx::broadcast<vidx_t>(simd_abi_traits<T>::size() + 1);
        auto dest = dx::select(mask, active, inactive);
        for (auto j = 1zu; j < simd_size; j <<= 1) {
            auto const vj = dx::broadcast<vidx_t>(__DPL to_signed(j));
            auto const shifted_dest = dx::slide_left(dest, max, j);
            auto const delta = dx::subtract(dx::add(iota, vj), shifted_dest);
            auto take = dx::cmpneq(dx::bwand(delta, vj), dx::zero);
            take = dx::bwandnot(take, dx::cmpgt(shifted_dest, iota));

            val = dx::shift_left(val, take, val, j);
            dest = dx::select(take, shifted_dest, dest);
        }

        return dx::select(dx::cmplt(iota, pop), val, src);
    }

    template <canonical_vector T>
    requires fixed_width_abi<simd_abi_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T val, simd_mask_type_t<T> mask, type_identity_t<T> src) noexcept {
        using A = simd_abi_type_t<T>;
        using vidx_t = signed_canonical_vector_t<T>;
        auto const vone = dx::broadcast<vidx_t>(dx::one);
        auto const pop = dx::popcount(mask);
        auto const active =
            dx::exscan_sum(dx::select(mask, vone, dx::zero), dx::zero);
        auto const inactive =
            dx::exscan_sum(dx::select(mask, dx::zero, vone), pop);
        auto const iota = dx::lane_index<vidx_t>();
        auto const max = dx::broadcast<vidx_t>(simd_abi_traits<T>::size() + 1);
        [&]<size_t J>(this auto self, auto dest, immediate<J> jmm) {
            auto const shifted_dest = dx::slide_left(dest, max, jmm);
            auto const vj = dx::broadcast<vidx_t>(jmm);
            auto const delta = dx::subtract(dx::add(iota, vj), shifted_dest);
            auto take = dx::cmpneq(dx::bwand(delta, vj), dx::zero);
            take = dx::bwandnot(take, dx::cmpgt(shifted_dest, iota));
            val = dx::shift_left(val, take, val, jmm);
            if constexpr (J * 2 < simd_abi_traits<T>::size()) {
                self(dx::select(take, shifted_dest, dest), imm<J * 2>);
            }
        }(dx::select(mask, active, inactive), imm<1zu>);

        return dx::select(dx::cmplt(iota, pop), val, src);
    }

    template <canonical_vector T, const_mask_for<T> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        T val, M mask, type_identity_t<T> src) noexcept {
        using A = simd_abi_type_t<T>;
        constexpr auto cmask = dx::to_const_mask<T>(mask);
        constexpr auto maskbits = dx::to_bitset(cmask);
        if constexpr (dx::none_of(cmask)) {
            return src;
        } else if constexpr (dx::all_of(cmask)) {
            return val;
        } else if constexpr (__DPL popcount(maskbits) ==
            __DPL countr_one(maskbits >> __DPL countr_zero(maskbits))) {
            // contiguous set bits
            constexpr auto shift = __DPL countr_zero(maskbits);
            constexpr auto V =
                (1zu << dx::popcount(dx::to_const_mask<T>(mask))) - 1;
            constexpr make_const_mask_t<T, V> new_mask{};

            return dx::shift_left(src, new_mask, val, imm<shift>);
        } else {
            constexpr auto iota = iota_sequence<T>;
            constexpr auto active = calc_active(cmask);
            constexpr auto inactive = calc_inactive(cmask);

            [&val, &iota]<size_t J>(
                this auto self, auto dest, immediate<J> jmm) {
                constexpr auto shifted_dest =
                    fallback_impl::template shift_left<J>(dest);
                constexpr auto take =
                    fallback_impl::template calc_take<J>(shifted_dest);

                val = dx::shift_left(val, cmask_v<take>, val, jmm);
                if constexpr (J * 2 < simd_abi_traits<T>::size()) {
                    constexpr auto new_dest = fallback_impl::select(
                        cmask_v<take>, shifted_dest, dest);
                    self(new_dest, imm<J * 2>);
                }
            }(fallback_impl::select(cmask, active, inactive), imm<1zu>);

            constexpr auto V =
                (1zu << dx::popcount(dx::to_const_mask<T>(mask))) - 1;
            constexpr make_const_mask_t<T, V> new_mask{};
            return dx::select(new_mask, val, src);
        }
    }

    template <simd_vector T, exact_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<compress_t, T, M, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<compress_t, T, M, canonical_type_t<T>>
    operator()(T&& val, M&& mask, V&& src) noexcept(
        canonical_mask<M> && canonical_vector<T>) {
        return compress_t::operator()(__DPL forward<T>(val),
            __DPL forward<M>(mask),
            dx::broadcast<T>(__DPL forward<V>(src)));
    }

    template <simd_vector T, const_mask_for<T> M, broadcastable_to<T> V>
    requires cpo_invocable<compress_t, T, M, canonical_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr cpo_result_t<compress_t, T, M, canonical_type_t<T>>
    operator()(T&& val, M mask, V&& src) noexcept(canonical_vector<T>) {
        return compress_t::operator()( __DPL forward<T>(val), mask,
            dx::broadcast<T>(__DPL forward<V>(src)));
    }

private:
    template <size_t W>
    static consteval size_t prefix_sum(bitset<W> set, size_t idx) noexcept {
        auto result = 0zu;
        for (auto i = 0zu; i < idx; ++i) {
            result += set.test(i);
        }
        return result;
    }

    template <size_t W, internal::mask_value_t<W> V>
    static consteval auto calc_active(const_mask<W, V> mask) noexcept {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            constexpr auto bits = dx::to_bitset(mask);
            return index_sequence<prefix_sum(bits, Is)...>{};
        }(make_index_sequence<W>{});
    }

    template <size_t W, internal::mask_value_t<W> V>
    static consteval auto calc_inactive(const_mask<W, V> mask) noexcept {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            constexpr auto bits = ~dx::to_bitset(mask);
            constexpr auto pop = __DPL popcount(bits);
            return index_sequence<pop + prefix_sum(bits, Is)...>{};
        }(make_index_sequence<W>{});
    }

    template <size_t... Is, size_t... Js>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto cmplt(
        index_sequence<Is...>, index_sequence<Js...>) noexcept {
        constexpr auto width = sizeof...(Is);
        return bitset<width>((Is < Js)...);
    }

    template <size_t... Is, size_t J>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto cmplt(index_sequence<Is...>, immediate<J>) noexcept {
        constexpr auto width = sizeof...(Is);
        return bitset<width>((Is < J)...);
    }

    template <size_t... Is, size_t... Js>
    static consteval auto select(
        auto cmask, index_sequence<Is...>, index_sequence<Js...>) noexcept {
        constexpr auto bits = dx::to_bitset(cmask);
        return [&]<size_t... Ks>(index_sequence<Ks...>) {
            return index_sequence<(bits[Ks] ? Is : Js)...>{};
        }(make_index_sequence<sizeof...(Is)>{});
    }

    template <size_t N, size_t... Is>
    static consteval auto shift_left(index_sequence<Is...>) noexcept {
        return []<size_t... Ks>(index_sequence<Ks...>) {
            constexpr auto size = sizeof...(Is);
            constexpr size_t array[]{Is...};
            return index_sequence<(
                Ks + N < size ? array[Ks + N] : size + 1)...>{};
        }(make_index_sequence<sizeof...(Is)>{});
    }

    template <size_t N, size_t... Is>
    static consteval auto calc_take(index_sequence<Is...>) noexcept {
        constexpr auto delta = [&]<size_t... Ks>(index_sequence<Ks...>) {
            return index_sequence<(Ks + N - Is)...>{};
        }(make_index_sequence<sizeof...(Is)>{});
        constexpr auto lhs = []<size_t... Ks>(index_sequence<Ks...>) {
            return bitset<sizeof...(Is)>(((Ks & N) != 0)...);
        }(delta);
        constexpr auto rhs = [&]<size_t... Ks>(index_sequence<Ks...>) {
            return bitset<sizeof...(Is)>((Is <= Ks)...);
        }(make_index_sequence<sizeof...(Is)>{});
        return lhs & rhs;
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::compress_t compress{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
