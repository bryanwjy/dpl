// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"
#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/broadcast.h"
#  include "dpl/xmm/basic/extract.h"
#  include "dpl/xmm/basic/initialize.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/immediate.h"
#    include "dpl/core/type_traits/representation.h"
#    include "dpl/std/bit/bit_cast.h"
#    include "dpl/std/type_traits/sequence.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
namespace internal {
template <simd_element E, integral I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(E const* ptr, vector<I> idx) noexcept {
    return [&]<size_t... Is>(index_sequence<Is...>) {
        constexpr auto size = sizeof...(Is);
        return xmm::initialize<E>([&]<size_t J>(immediate<J> imm) {
            if constexpr (J < vector<I>::size()) {
                return ptr[xmm::extract(idx, imm)];
            } else {
                constexpr E zero{};
                return zero;
            }
        }(imm<Is>)...);
    }(iota<E>);
}

template <simd_element E, integral I>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(type_identity_t<vector<E>> src,
        type_identity_t<mask<E>> mask, E const* ptr, vector<I> idx) noexcept {
    return [&]<size_t... Is>(index_sequence<Is...>) {
        constexpr auto size = sizeof...(Is);
        return xmm::initialize<E>([&]<size_t J>(immediate<J> imm) {
            if constexpr (J < vector<I>::size()) {
                return mask[Is] ? ptr[xmm::extract(idx, imm)] : src[Is];
            } else {
                constexpr E zero{};
                return zero;
            }
        }(imm<Is>)...);
    }(iota<E>);
}

template <simd_element E, integral I, xmm::imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(type_identity_t<vector<E>> src,
        xmm::cmask_t<E, M> mask, E const* ptr, vector<I> idx) noexcept {
    return [&]<size_t... Is>(index_sequence<Is...>) {
        constexpr auto size = sizeof...(Is);
        return xmm::initialize<E>([&]<size_t J>(immediate<J> imm) {
            if constexpr (J < vector<I>::size()) {
                return mask[Is] ? ptr[xmm::extract(idx, imm)] : src[Is];
            } else {
                constexpr E zero{};
                return zero;
            }
        }(imm<Is>)...);
    }(iota<E>);
}
} // namespace internal

#  if DPL_SIMD_X86_AVX2

template <simd_element E>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(E const* ptr, vector<int32> idx) noexcept {
    if consteval {
        return internal::gather(ptr, idx);
    } else {
        if constexpr (same_as<E, int32>) {
            return _mm_i32gather_epi32(ptr, +idx, sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<float>>) {
            return _mm_i32gather_ps(ptr, +idx, sizeof(E));
        } else if constexpr (same_as<E, int64>) {
            return _mm_i32gather_epi64(ptr, +idx, sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<double>>) {
            return _mm_i32gather_pd(ptr, +idx, sizeof(E));
        } else {
            using rep_t = dx::signed_representation_t<E>;
            auto const* data = reinterpret_cast<rep_t const*>(ptr);
            if constexpr (same_as<native_vector_t<rep_t>, native_vector_t<E>>) {
                return +xmm::gather(data, idx);
            } else {
                return __DPL bit_cast<native_vector_t<E>>(
                    +xmm::gather(data, idx));
            }
        }
    }
}

template <simd_element E>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(type_identity_t<vector<E>> src,
        type_identity_t<mask<E>> mask, E const* ptr,
        vector<int32> idx) noexcept {
    if consteval {
        return internal::gather(src, mask, ptr, idx);
    } else {
        if constexpr (same_as<E, int32>) {
            return _mm_mask_i32gather_epi32(+src, ptr, +idx, +mask, sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<float>>) {
            return _mm_mask_i32gather_ps(+src, ptr, +idx, +mask, sizeof(E));
        } else if constexpr (same_as<E, int64>) {
            return _mm_mask_i32gather_epi64(+src, ptr, +idx, +mask, sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<double>>) {
            return _mm_mask_i32gather_pd(+src, ptr, +idx, +mask, sizeof(E));
        } else {
            using rep_t = dx::signed_representation_t<E>;
            auto const* data = reinterpret_cast<rep_t const*>(ptr);
            auto const rep = [&]() -> vector<rep_t> {
                if constexpr (same_as<native_vector_t<E>, __m128i>) {
                    return +src;
                } else {
                    return __DPL bit_cast<native_vector_t<E>>(+src);
                }
            }();
            if constexpr (same_as<native_vector_t<rep_t>, native_vector_t<E>>) {
                static_assert(is_convertible_v<xmm::mask<E>, xmm::mask<rep_t>>);
                return +xmm::gather(rep, mask, data, idx);
            } else {
                return __DPL bit_cast<native_vector_t<E>>(
                    +xmm::gather(rep, mask, data, idx));
            }
        }
    }
}

template <simd_element E>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(dx::zero_t zero, type_identity_t<mask<E>> mask,
    E const* ptr, vector<int32> idx) noexcept {
    return xmm::gather(xmm::broadcast<E>(zero), mask, ptr, idx);
}

template <simd_element E>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(E const* ptr, vector<int64> idx) noexcept {
    if consteval {
        return internal::gather(ptr, idx);
    } else {
        if constexpr (same_as<E, int64>) {
            return _mm_i64gather_epi64(ptr, +idx, sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<double>>) {
            return _mm_i64gather_pd(ptr, +idx, sizeof(E));
        } else if constexpr (same_as<E, int32>) {
            return _mm_i64gather_epi32(ptr, +idx, sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<float>>) {
            return _mm_i64gather_ps(ptr, +idx, sizeof(E));
        } else {
            using rep_t = dx::signed_representation_t<E>;
            auto const* data = reinterpret_cast<rep_t const*>(ptr);
            if constexpr (same_as<native_vector_t<rep_t>, native_vector_t<E>>) {
                return +xmm::gather(data, idx);
            } else {
                return __DPL bit_cast<native_vector_t<E>>(
                    +xmm::gather(data, idx));
            }
        }
    }
}

template <simd_element E>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(type_identity_t<vector<E>> src,
        type_identity_t<mask<E>> mask, E const* ptr,
        vector<int64> idx) noexcept {
    if consteval {
        return internal::gather(src, mask, ptr, idx);
    } else {
        if constexpr (same_as<E, int32>) {
            return _mm_mask_i64gather_epi32(+src, ptr, +idx, +mask, sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<float>>) {
            return _mm_mask_i64gather_ps(+src, ptr, +idx, +mask, sizeof(E));
        } else if constexpr (same_as<E, int64>) {
            return _mm_mask_i64gather_epi64(+src, ptr, +idx, +mask, sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<double>>) {
            return _mm_mask_i64gather_pd(+src, ptr, +idx, +mask, sizeof(E));
        } else {
            using rep_t = dx::signed_representation_t<E>;
            auto const* data = reinterpret_cast<rep_t const*>(ptr);
            auto const rep = [&]() -> vector<rep_t> {
                if constexpr (same_as<native_vector_t<E>, __m128i>) {
                    return +src;
                } else {
                    return __DPL bit_cast<native_vector_t<E>>(+src);
                }
            }();
            if constexpr (same_as<native_vector_t<rep_t>, native_vector_t<E>>) {
                if constexpr (same_as<unsigned int, E>) {
                    static_assert(same_as<decltype(rep), vector<int> const>);
                    static_assert(same_as<decltype(data), int const*>);
                    static_assert(requires(
                        void (*f)(xmm::mask<rep_t>), xmm::mask<E> m) { f(m); });
                }

                return +xmm::gather(rep, mask, data, idx);
            } else {
                return __DPL bit_cast<native_vector_t<E>>(
                    +xmm::gather(rep, mask, data, idx));
            }
        }
    }
}

template <simd_element E>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(dx::zero_t zero, type_identity_t<mask<E>> mask,
    E const* ptr, vector<int64> idx) noexcept {
    return xmm::gather(xmm::broadcast<E>(zero), mask, ptr, idx);
}

#    if DPL_SIMD_x86_AVX512F & DPL_SIMD_x86_AVX512VL

template <simd_element E, xmm::imask_t<E> M>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(type_identity_t<vector<E>> src,
        xmm::cmask_t<E, M> mask, E const* ptr, vector<int32> idx) noexcept {
    if consteval {
        return internal::gather(src, mask, ptr, idx);
    } else {
        if constexpr (same_as<E, int32>) {
            return _mm_mmask_i32gather_epi32(
                +src, ptr, +idx, static_cast<__mmask8>(M), sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<float>>) {
            return _mm_mmask_i32gather_ps(
                +src, ptr, +idx, static_cast<__mmask8>(M), sizeof(E));
        } else if constexpr (same_as<E, int64>) {
            return _mm_mmask_i32gather_epi64(
                +src, ptr, +idx, static_cast<__mmask8>(M), sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<double>>) {
            return _mm_mmask_i32gather_pd(
                +src, ptr, +idx, static_cast<__mmask8>(M), sizeof(E));
        } else {
            using rep_t = dx::signed_representation_t<E>;
            auto const* data = reinterpret_cast<rep_t const*>(ptr);
            auto const rep = [&]() -> vector<rep_t> {
                if constexpr (same_as<native_vector_t<E>, __m128i>) {
                    return +src;
                } else {
                    return __DPL bit_cast<native_vector_t<E>>(+src);
                }
            }();
            if constexpr (same_as<native_vector_t<rep_t>, native_vector_t<E>>) {
                return +xmm::gather(rep, mask, data, idx);
            } else {
                return __DPL bit_cast<native_vector_t<E>>(
                    +xmm::gather(rep, mask, data, idx));
            }
        }
    }
}

template <simd_element E, xmm::imask_t<E> M>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(dx::zero_t zero, xmm::cmask_t<E, M> mask,
    E const* ptr, vector<int32> idx) noexcept {
    return xmm::gather(xmm::broadcast<E>(zero), mask, ptr, idx);
}

template <simd_element E, xmm::imask_t<E> M>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
constexpr vector<E>
    DPL_VECTORCALL gather(type_identity_t<vector<E>> src,
        xmm::cmask_t<E, M> mask, E const* ptr, vector<int64> idx) noexcept {
    if consteval {
        return internal::gather(src, mask, ptr, idx);
    } else {
        if constexpr (same_as<E, int32>) {
            return _mm_mmask_i64gather_epi32(
                +src, ptr, +idx, static_cast<__mmask8>(M), sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<float>>) {
            return _mm_mmask_i64gather_ps(
                +src, ptr, +idx, static_cast<__mmask8>(M), sizeof(E));
        } else if constexpr (same_as<E, int64>) {
            return _mm_mmask_i64gather_epi64(
                +src, ptr, +idx, static_cast<__mmask8>(M), sizeof(E));
        } else if constexpr (same_as<native_vector_t<E>,
                                 native_vector_t<double>>) {
            return _mm_mmask_i64gather_pd(
                +src, ptr, +idx, static_cast<__mmask8>(M), sizeof(E));
        } else {
            using rep_t = dx::signed_representation_t<E>;
            auto const* data = reinterpret_cast<rep_t const*>(ptr);
            auto const rep = [&]() -> vector<rep_t> {
                if constexpr (same_as<native_vector_t<E>, __m128i>) {
                    return +src;
                } else {
                    return __DPL bit_cast<native_vector_t<E>>(+src);
                }
            }();
            if constexpr (same_as<native_vector_t<rep_t>, native_vector_t<E>>) {
                return +xmm::gather(rep, mask, data, idx);
            } else {
                return __DPL bit_cast<native_vector_t<E>>(
                    +xmm::gather(rep, mask, data, idx));
            }
        }
    }
}

template <simd_element E, xmm::imask_t<E> M>
requires common_size_with<E, int32> || common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(dx::zero_t zero, xmm::cmask_t<E, M> mask,
    E const* ptr, vector<int64> idx) noexcept {
    return xmm::gather(xmm::broadcast<E>(zero), mask, ptr, idx);
}

template <simd_element E, xmm::imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, dx::zero_t zero, xmm::cmask_t<E, M> mask,
    E const* ptr, vector<int32> idx) noexcept
requires requires { xmm::gather(zero, mask, ptr, idx); }
{
    return xmm::gather(zero, mask, ptr, idx);
}

template <simd_element E, xmm::imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, type_identity_t<vector<E>> src,
    xmm::cmask_t<E, M> mask, E const* ptr, vector<int32> idx) noexcept
requires requires { xmm::gather(src, mask, ptr, idx); }
{
    return xmm::gather(src, mask, ptr, idx);
}

template <simd_element E, xmm::imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, dx::zero_t zero, xmm::cmask_t<E, M> mask,
    E const* ptr, vector<int64> idx) noexcept
requires requires { xmm::gather(zero, mask, ptr, idx); }
{
    return xmm::gather(zero, mask, ptr, idx);
}

template <simd_element E, xmm::imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, type_identity_t<vector<E>> src,
    xmm::cmask_t<E, M> mask, E const* ptr, vector<int64> idx) noexcept
requires requires { xmm::gather(src, mask, ptr, idx); }
{
    return xmm::gather(src, mask, ptr, idx);
}

#    endif //  #if DPL_SIMD_x86_AVX512F & DPL_SIMD_x86_AVX512VL
#  endif   //  #if DPL_SIMD_x86_AVX2

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, E const* ptr, vector<int32> idx) noexcept
requires requires { xmm::gather(ptr, idx); }
{
    return xmm::gather(ptr, idx);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, dx::zero_t zero,
    type_identity_t<mask<E>> mask, E const* ptr, vector<int32> idx) noexcept
requires requires { xmm::gather(zero, mask, ptr, idx); }
{
    return xmm::gather(zero, mask, ptr, idx);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, type_identity_t<vector<E>> src,
    type_identity_t<mask<E>> mask, E const* ptr, vector<int32> idx) noexcept
requires requires { xmm::gather(src, mask, ptr, idx); }
{
    return xmm::gather(src, mask, ptr, idx);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, E const* ptr, vector<int64> idx) noexcept
requires requires { xmm::gather(ptr, idx); }
{
    return xmm::gather(ptr, idx);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, dx::zero_t zero,
    type_identity_t<mask<E>> mask, E const* ptr, vector<int64> idx) noexcept
requires requires { xmm::gather(zero, mask, ptr, idx); }
{
    return xmm::gather(zero, mask, ptr, idx);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> gather(abi_tag, type_identity_t<vector<E>> src,
    type_identity_t<mask<E>> mask, E const* ptr, vector<int64> idx) noexcept
requires requires { xmm::gather(src, mask, ptr, idx); }
{
    return xmm::gather(src, mask, ptr, idx);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
