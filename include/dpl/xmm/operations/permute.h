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

#    include "dpl/core/numbers/integral_traits.h"
#    include "dpl/std/bit/char_bit.h"
#    include "dpl/std/concepts/tuple_like.h"
#    include "dpl/std/type_traits/conditional.h" // IWYU pragma: keep
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
consteval int pack_indices(index_sequence<Is...> = {}) noexcept {
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
    } else if constexpr (common_size_with<E, double>) {
        return xmm::reinterpret<E>(xmm::shuffle(
            xmm::reinterpret<double>(lhs), xmm::reinterpret<double>(rhs), seq));
    } else {
        static_assert(common_size_with<E, float>);
        return xmm::reinterpret<E>(xmm::shuffle(
            xmm::reinterpret<float>(lhs), xmm::reinterpret<float>(rhs), seq));
    }
}

template <size_t I0, size_t I1, size_t I2, size_t I3>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL shuffle(
        vector<int32> val, index_sequence<I0, I1, I2, I3> seq = {}) noexcept {
    return _mm_shuffle_epi32(+val, details::pack_indices(seq));
}

template <size_t... Is, simd_element E>
requires (sizeof...(Is) == vector<E>::size())
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(
        vector<E> val, index_sequence<Is...> seq = {}) noexcept {
    if constexpr (is_same_v<index_sequence<Is...>,
                      make_index_sequence<sizeof...(Is)>>) {
        return val;
    } else if constexpr (requires { xmm::shuffle(val, seq); }) {
        return xmm::shuffle(val, seq);
    } else if constexpr (requires { xmm::shuffle(val, val, seq); }) {
        return xmm::shuffle(val, val, seq);
    } else if constexpr (sizeof(E) == 1) {
        if constexpr (!is_same_v<int8, E>) {
            return xmm::reinterpret<E>(
                xmm::permute(xmm::reinterpret<int8>(val), seq));
        } else {
            constexpr auto array = details::idx_array<Is...>;
#  if DPL_SIMD_X86_SSE3
            return _mm_shuffle_epi8(+val, +xmm::load(array));
#  else
            constexpr auto getlo = [&](size_t j) consteval {
                return array[2 * j] >> 1;
            };
            constexpr auto gethi = [&](size_t j) consteval {
                return array[2 * j + 1] >> 1;
            };
            constexpr bool word_aligned = [&] {
                for (int j = 0; j < 8; ++j) {
                    if ((array[2 * j] & 1) != 0 ||
                        array[2 * j + 1] != array[2 * j] + 1) {
                        return false;
                    }
                }
                return true;
            }();
            if constexpr (word_aligned) {
                return xmm::permute<getlo(0), getlo(1), getlo(2), getlo(3),
                    getlo(4), getlo(5), getlo(6), getlo(7)>(
                    xmm::reinterpret<int16>(val));
            } else {
                constexpr auto lobyte = static_cast<int16>(0xff);
                constexpr auto hibyte = static_cast<int16>(0xff00);

                auto const lomask = [&]<size_t... Js>(index_sequence<Js...>) {
                    alignas(xmm::abi_tag::alignment) constexpr int16 mask[] = {
                        static_cast<int16>(
                            (array[2 * Js] & 1) == 0 ? lobyte : 0)...};
                    return +xmm::load(dx::aligned, mask);
                }(iota<int16>);
                auto const himask = [&]<size_t... Js>(index_sequence<Js...>) {
                    alignas(xmm::abi_tag::alignment) constexpr int16 mask[] = {
                        static_cast<int16>(
                            (array[2 * Js + 1] & 1) == 1 ? hibyte : 0)...};
                    return +xmm::load(dx::aligned, mask);
                }(iota<int16>);

                auto const lo = xmm::permute<getlo(0), getlo(1), getlo(2),
                    getlo(3), getlo(4), getlo(5), getlo(6), getlo(7)>(
                    xmm::reinterpret<int16>(val));
                auto const hi = xmm::permute<gethi(0), gethi(1), gethi(2),
                    gethi(3), gethi(4), gethi(5), gethi(6), gethi(7)>(
                    xmm::reinterpret<int16>(val));

                auto const vlobyte = +xmm::broadcast<int16>(lobyte);
                auto const vhibyte = +xmm::broadcast<int16>(hibyte);
                auto vlo = _mm_srli_epi16(+lo, 8);
                vlo = _mm_and_si128(vlo, _mm_andnot_si128(lomask, vlobyte));
                vlo = _mm_or_si128(vlo, _mm_and_si128(+lo, lomask));
                auto vhi = _mm_slli_epi16(+hi, 8);
                vhi = _mm_and_si128(vhi, _mm_andnot_si128(himask, vhibyte));
                vhi = _mm_or_si128(vhi, _mm_and_si128(+hi, himask));
                return _mm_or_si128(vlo, vhi);
            }
#  endif
        }
    } else {
        static_assert(sizeof(E) == 2);
        if constexpr (!is_same_v<int16, E>) {
            return xmm::reinterpret<E>(
                xmm::permute(xmm::reinterpret<int16>(val), seq));
        } else {
            constexpr auto array = details::idx_array<Is...>;
#  if DPL_SIMD_X86_SSE3
            return [&]<size_t... Js>(index_sequence<Js...>) {
                return xmm::reinterpret<E>(
                    xmm::permute<(array[Js / 2] * 2 + (Js % 2))...>(
                        xmm::reinterpret<int8>(val)));
            }(make_index_sequence<sizeof...(Is) * 2>{});
#  else
            constexpr auto imm = details::pack_indices(seq);
            constexpr auto imm_lo = [&]<size_t... Js>(index_sequence<Js...>) {
                return (... | ((array[Js] & 3) << (Js * 2)));
            }(make_index_sequence<4>{});
            constexpr auto imm_hi = [&]<size_t... Js>(index_sequence<Js...>) {
                return (... | ((array[4 + Js] & 3) << (Js * 2)));
            }(make_index_sequence<4>{});

            constexpr bool cross_lanes = [&] {
                for (auto j = 0; j < 8; ++j) {
                    if ((array[j] < 4) != (j < 4)) {
                        return true;
                    }
                }

                return false;
            }();

            auto cx =
                _mm_shufflehi_epi16(_mm_shufflelo_epi16(+val, imm_lo), imm_hi);

            if constexpr (!cross_lanes) {
                return cx;
            } else {
                constexpr auto imm = details::pack_indices<2, 3, 0, 1>();
                auto y = _mm_shuffle_epi32(+val, imm);
                auto cy =
                    _mm_shufflehi_epi16(_mm_shufflelo_epi16(y, imm_lo), imm_hi);
                auto const vcond = [&]<size_t... Js>(index_sequence<Js...>) {
                    alignas(xmm::abi_tag::alignment) constexpr int16 mask[] = {
                        static_cast<int16>(
                            (array[Js] < 4) == (Js < 4) ? -1 : 0)...};
                    return +xmm::load(dx::aligned, mask);
                }(iota<int16>);

                return _mm_or_si128(
                    _mm_and_si128(vcond, cx), _mm_andnot_si128(vcond, cy));
            }
#  endif
        }
    }
}

#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
template <size_t... Is, common_size_with<float> E, imask_t<E> M>
requires (sizeof...(Is) == vector<E>::size())
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(dx::zero_t zero, cmask_t<E, M> mask, vector<E> val,
        index_sequence<Is...> seq) noexcept {
    if constexpr (is_same_v<native_vector_t<E>, __m128>) {
        return _mm_maskz_shuffle_ps(M, +val, +val, details::pack_indices(seq));
    } else {
        return _mm_maskz_shuffle_epi32(
            M, +xmm::reinterpret<int32>(val), details::pack_indices(seq));
    }
}

template <size_t... Is, common_size_with<float> E, imask_t<E> M>
requires (sizeof...(Is) == vector<E>::size())
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(type_identity_t<vector<E>> src, cmask_t<E, M> mask,
        vector<E> val, index_sequence<Is...> seq) noexcept {
    if constexpr (is_same_v<native_vector_t<E>, __m128>) {
        return _mm_mask_shuffle_ps(
            +src, M, +val, +val, details::pack_indices(seq));
    } else {
        return _mm_mask_shuffle_epi32(
            +src, M, +val, details::pack_indices(seq));
    }
}

template <size_t... Is, common_size_with<double> E, imask_t<E> M>
requires (sizeof...(Is) == vector<E>::size())
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(dx::zero_t zero, cmask_t<E, M> mask, vector<E> val,
        index_sequence<Is...> seq) noexcept {
    if constexpr (is_same_v<native_vector_t<E>, __m128d>) {
        return _mm_maskz_shuffle_pd(M, +val, +val, details::pack_indices(seq));
    } else {
        return xmm::reinterpret<E>(
            xmm::permute(zero, mask, xmm::reinterpret<double>(val), seq));
    }
}

template <size_t... Is, common_size_with<double> E, imask_t<E> M>
requires (sizeof...(Is) == vector<E>::size())
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(type_identity_t<vector<E>> src, cmask_t<E, M> mask,
        vector<E> val, index_sequence<Is...> seq) noexcept {
    if constexpr (is_same_v<native_vector_t<E>, __m128d>) {
        return _mm_mask_shuffle_pd(
            +src, M, +val, +val, details::pack_indices(seq));
    } else {
        return xmm::reinterpret<E>(xmm::permute(xmm::reinterpret<double>(src),
            mask, xmm::reinterpret<double>(val), seq));
    }
}
#  endif // if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL

#  if DPL_SIMD_X86_SSE3
template <template_barrier_t = template_barrier, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(vector<E> val, ssize_vector_t<E> seq) noexcept {
    if constexpr (!integral<E>) {
        using sint_t = signed_representation_t<E>;
        return xmm::reinterpret<E>(
            xmm::permute(xmm::reinterpret<sint_t>(val), seq));
    } else if constexpr (sizeof(E) == 1) {
        return _mm_shuffle_epi8(+val, +seq);
    } else if constexpr (sizeof(E) == 2) {
        seq = _mm_or_si128(_mm_slli_epi16(+seq, 9), _mm_slli_epi16(+seq, 1));
        seq = _mm_add_epi16(+seq, +xmm::broadcast<int16>(0x100));
        return _mm_shuffle_epi8(+val, +seq);
    } else if constexpr (sizeof(E) == 4) {
        seq = _mm_slli_epi16(+seq, 2);
        constexpr auto seq8 = details::widen_iota<4, int32>();
        seq = xmm::reinterpret<int32>(
            xmm::permute(xmm::reinterpret<int8>(seq), seq8));
        seq = xmm::add(seq, xmm::broadcast<int32>(0x03020100));
        return _mm_shuffle_epi8(+val, +seq);
    } else {
        static_assert(sizeof(E) == 8);
        auto const lo = vector<E>(_mm_unpacklo_epi64(+val, +val));
        auto const hi = vector<E>(_mm_unpackhi_epi64(+val, +val));
        return xmm::select(xmm::cmpeq(seq, dx::zero), lo, hi);
    }
}
#  endif // if DPL_SIMD_X86_SSE3

#  if DPL_SIMD_X86_AVX
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL permute(
    vector<float> val, ssize_vector_t<float> seq) noexcept {
    return _mm_permutevar_ps(+val, +seq);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double> DPL_VECTORCALL permute(
    vector<double> val, ssize_vector_t<double> seq) noexcept {
    return _mm_permutevar_pd(+val, _mm_slli_epi64(+seq, 1));
}

template <template_barrier_t = template_barrier, simd_element E>
requires (sizeof(E) == sizeof(float) || sizeof(E) == sizeof(double))
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(vector<E> val, ssize_vector_t<E> seq) noexcept {
    using float_t DPL_NODEBUG =
        conditional_t<sizeof(E) == sizeof(float), float, double>;
    return xmm::reinterpret<E>(
        xmm::permute(xmm::reinterpret<float_t>(val), seq));
}
#  endif // if DPL_SIMD_X86_AVX

#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
template <template_barrier_t = template_barrier, common_size_with<float> E,
    imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(dx::zero_t zero, cmask_t<E, M> mask, vector<E> val,
        ssize_vector_t<E> seq) noexcept {
    return _mm_maskz_permutevar_ps(M, +val, +seq);
}

template <template_barrier_t = template_barrier, common_size_with<float> E,
    imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(type_identity_t<vector<E>> src, cmask_t<E, M> mask,
        vector<E> val, ssize_vector_t<E> seq) noexcept {
    if constexpr (integral<E>) {
        return xmm::reinterpret<E>(xmm::permute(xmm::reinterpret<float>(src),
            mask, xmm::reinterpret<float>(val), seq));
    } else {
        return _mm_mask_permutevar_ps(+src, M, +val, +seq);
    }
}

template <template_barrier_t = template_barrier, common_size_with<double> E,
    imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(dx::zero_t zero, cmask_t<E, M> mask, vector<E> val,
        ssize_vector_t<E> seq) noexcept {
    return _mm_maskz_permutevar_pd(M, +val, +seq);
}

template <template_barrier_t = template_barrier, common_size_with<double> E,
    imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL permute(type_identity_t<vector<E>> src, cmask_t<E, M> mask,
        vector<E> val, ssize_vector_t<E> seq) noexcept {
    if constexpr (integral<E>) {
        return xmm::reinterpret<E>(xmm::permute(xmm::reinterpret<double>(src),
            mask, xmm::reinterpret<double>(val), seq));
    } else {
        return _mm_mask_permutevar_pd(+src, M, +val, +seq);
    }
}
#  endif // if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL

template <size_t... Is, simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(
    abi_tag, vector<E> lhs, index_sequence<Is...> rhs = {}) noexcept
requires requires { xmm::permute(lhs, rhs); }
{
    return xmm::permute(lhs, rhs);
}

template <size_t... Is, simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, index_sequence<Is...> rhs = {}) noexcept
requires requires { xmm::permute(src, mask, lhs, rhs); }
{
    return xmm::permute(src, mask, lhs, rhs);
}

template <typename E>
concept permutable =
    requires(vector<E> lhs, ssize_vector_t<E> rhs) { xmm::permute(lhs, rhs); };

template <typename E, typename S, typename M>
concept impermutable = requires(S src, M mask, vector<E> lhs,
    ssize_vector_t<E> rhs) { xmm::permute(src, mask, lhs, rhs); };

template <template_barrier_t = template_barrier, simd_element E>
requires permutable<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(
    abi_tag, vector<E> lhs, ssize_vector_t<E> rhs) noexcept {
    return xmm::permute(lhs, rhs);
}

template <template_barrier_t = template_barrier, simd_element E, imask_t<E> M>
requires impermutable<E, vector<E>, cmask_t<E, M>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, ssize_vector_t<E> rhs) noexcept {
    return xmm::permute(src, mask, lhs, rhs);
}

template <template_barrier_t = template_barrier, simd_element E, imask_t<E> M>
requires impermutable<E, dx::zero_t, cmask_t<E, M>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> lhs, ssize_vector_t<E> rhs) noexcept {
    return xmm::permute(zero, mask, lhs, rhs);
}

template <template_barrier_t = template_barrier, simd_element E,
    unsigned_integral I>
requires permutable<E> && common_size_with<E, I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(abi_tag, vector<E> lhs, vector<I> rhs) noexcept {
    return xmm::permute(lhs, +rhs);
}

template <template_barrier_t = template_barrier, simd_element E,
    unsigned_integral I, imask_t<E> M>
requires impermutable<E, vector<E>, cmask_t<E, M>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, vector<I> rhs) noexcept {
    return xmm::permute(src, mask, lhs, +rhs);
}

template <template_barrier_t = template_barrier, simd_element E,
    unsigned_integral I, imask_t<E> M>
requires impermutable<E, dx::zero_t, cmask_t<E, M>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> permute(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> lhs, vector<I> rhs) noexcept {
    return xmm::permute(zero, mask, lhs, +rhs);
}
} // namespace datapar::xmm
__DPL_DEFAULT_NAMESPACE_END

#endif
