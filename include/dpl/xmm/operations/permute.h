// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/arithmetic.h"
#  include "dpl/xmm/operations/compare.h"
#  include "dpl/xmm/operations/reinterpret.h"
#  include "dpl/xmm/operations/select.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/std/bit/char_bit.h"
#    include "dpl/std/concepts/tuple_like.h"
#    include "dpl/std/type_traits/sequence.h"
#    include "dpl/std/type_traits/signed_integral_type.h"
#    include "dpl/std/utility/template_barrier.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"
#    include "dpl/xmm/basic/load.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::xmm {
namespace details {
template <size_t... Is>
consteval int pack_indices(index_sequence<Is...>) noexcept {
    return []<size_t... Js>(index_sequence<Js...>) {
        constexpr auto width = __DPL bit_width(sizeof...(Is)) - 1;
        return (... | (Is << (Js * width)));
    }(make_index_sequence<sizeof...(Is)>{});
}

template <size_t... Is>
alignas(xmm::abi_tag::alignment) inline constexpr signed_integral_type_t<
    xmm::abi_tag::size* __DPL char_bit_v /
    sizeof...(Is)> idx_array[sizeof...(Is)]{Is...};

template <size_t N, simd_element E>
consteval auto widen_iota() noexcept {
    return []<size_t... Is>(index_sequence<Is...>) {
        return index_sequence<((Is / N) * N)...>{};
    }(make_index_sequence<vector<E>::size() * N>());
}
} // namespace details

template <size_t... Is, simd_element E>
requires (sizeof...(Is) == vector<E>::size()) &&
    (common_size_with<E, float> || common_size_with<E, double>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL shuffle(vector<E> lhs, type_identity_t<vector<E>> rhs,
        index_sequence<Is...> seq = {}) noexcept {
    if constexpr (is_same_v<E, float>) {
        return _mm_shuffle_ps(+lhs, +rhs, details::pack_indices(seq));
    } else if constexpr (is_same_v<E, double>) {
        return _mm_shuffle_pd(+lhs, +rhs, details::pack_indices(seq));
    } else if constexpr (common_size_with<E, float>) {
        return xmm::reinterpret<E>(xmm::shuffle(
            xmm::reinterpret<float>(lhs), xmm::reinterpret<float>(rhs), seq));
    } else {
        static_assert(common_size_with<E, double>);
        return xmm::reinterpret<E>(xmm::shuffle(
            xmm::reinterpret<double>(lhs), xmm::reinterpret<double>(rhs), seq));
    }
}

#  if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#  endif

template <size_t... Is, simd_element E>
requires (sizeof...(Is) == vector<E>::size())
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(
        vector<E> val, index_sequence<Is...> seq = {}) noexcept {
    if constexpr (is_same_v<index_sequence<Is...>,
                      make_index_sequence<sizeof...(Is)>>) {
        return val;
    } else if constexpr (requires { xmm::shuffle(val, val, seq); }) {
        return xmm::shuffle(val, val, seq);
    } else if constexpr (sizeof(E) == 1) {
        return _mm_shuffle_epi8(+xmm::reinterpret<int8>(val),
            +xmm::load(details::idx_array<Is...>));
    } else {
        static_assert(sizeof(E) == 2);

        return [&]<size_t... Js>(index_sequence<Js...>) {
#  if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && __cpp_pack_indexing >= 202311L
            return xmm::reinterpret<E>(
                xmm::permute<(Is...[Js / 2] * 2 + (Js % 2))...>(
                    xmm::reinterpret<int8>(val)));
#  else
            using iseq_t = index_sequence<Is...>;

            return xmm::reinterpret<E>(
                xmm::permute<( __DPL get<Js / 2>(iseq_t{}) * 2 + (Js % 2))...>(
                    xmm::reinterpret<int8>(val)));
#  endif
        }(make_index_sequence<sizeof...(Is) * 2>{});
    }
}

#  if DPL_SIMD_X86_AVX
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL permute(
    vector<float> val, ssize_vector_t<float> seq) noexcept {
    auto const inbound = xmm::cmplt(seq, xmm::broadcast<int32>(4));
    return xmm::select(
        inbound, vector<float>(_mm_permutevar_ps(+val, +seq)), dx::zero);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double> DPL_VECTORCALL permute(
    vector<double> val, ssize_vector_t<double> seq) noexcept {
    auto const inbound = xmm::cmplt(seq, xmm::broadcast<int64>(2));
    return xmm::select(inbound,
        vector<double>(_mm_permutevar_pd(+val, _mm_slli_epi64(+seq, 1))),
        dx::zero);
}
#  endif

template <template_barrier_t = template_barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(vector<E> val, ssize_vector_t<E> seq) noexcept {
    if constexpr (!integral<E>) {
        using sint_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::permute(xmm::reinterpret<sint_t>(val), seq));
    } else if constexpr (sizeof(E) == 1) {
        seq = xmm::select(
            xmm::cmplt(xmm::reinterpret<uint8>(seq), xmm::broadcast<uint8>(16)),
            seq, dx::all_bits);
        return _mm_shuffle_epi8(+val, +seq);
    } else if constexpr (sizeof(E) == 2) {
        auto lo = _mm_slli_epi16(+seq, 1);
        auto const inbound = xmm::cmplt(
            xmm::reinterpret<uint16>(seq), xmm::broadcast<uint16>(8));
        constexpr auto seq8 = details::widen_iota<2, int16>();
        auto vseq =
            xmm::reinterpret<int16>(xmm::permute(vector<int8>(lo), seq8));
        vseq = xmm::add(vseq, xmm::broadcast<int16>(0x100));
        vseq = xmm::select(inbound, vseq, dx::all_bits);
        return _mm_shuffle_epi8(+val, +vseq);
    } else if constexpr (sizeof(E) == 4) {
#  if DPL_SIMD_X86_AVX
        return xmm::reinterpret<E>(
            xmm::permute(xmm::reinterpret<float>(val), seq));
#  else
        auto lo = _mm_slli_epi16(+seq, 2);
        auto const inbound = xmm::cmplt(
            xmm::reinterpret<uint32>(seq), xmm::broadcast<uint32>(4));
        constexpr auto seq8 = details::widen_iota<4, int32>();
        auto vseq =
            xmm::reinterpret<int32>(xmm::permute(vector<int8>(lo), seq8));
        vseq = xmm::add(vseq, xmm::broadcast<int32>(0x03020100));
        vseq = xmm::select(inbound, vseq, dx::all_bits);
        return _mm_shuffle_epi8(+val, +vseq);
#  endif
    } else {
        static_assert(sizeof(E) == 8);
#  if DPL_SIMD_X86_AVX
        return xmm::reinterpret<E>(
            xmm::permute(xmm::reinterpret<double>(val), seq));
#  else
        auto const lo = vector<E>(_mm_unpacklo_epi64(+val, +val));
        auto const hi = vector<E>(_mm_unpackhi_epi64(+val, +val));
        auto const inbound = xmm::cmplt(
            xmm::reinterpret<uint64>(seq), xmm::broadcast<uint64>(2));
        auto result = xmm::select(
            xmm::cmpeq(seq, dx::broadcast<int64>(dx::zero)), lo, hi);
        return xmm::select(vector<E>(inbound), result, dx::zero);
#  endif
    }
}

#  if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#  endif

template <template_barrier_t = template_barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(abi_tag, vector<E> lhs, ssize_vector_t<E> rhs) noexcept
requires requires { xmm::permute(lhs, rhs); }
{
    return xmm::permute(lhs, rhs);
}

template <size_t... Is, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(
    abi_tag, vector<E> lhs, index_sequence<Is...> rhs = {}) noexcept
requires requires { xmm::permute(lhs, rhs); }
{
    return xmm::permute(lhs, rhs);
}

} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END

#endif
