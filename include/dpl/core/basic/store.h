// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/aligned.h"
#include "dpl/core/basic/extract.h"
#include "dpl/core/basic/immediate.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/array_for.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/bit_type.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void store(...) noexcept = delete;
void aligned_store(...) noexcept = delete;

template <typename T>
concept storable_simd =
    basic_simd_type<T> && requires(T src, typename T::value_type* dst) {
        store(internal::abi<T>, src, dst);
    };

template <typename S>
concept aligned_storable_simd =
    basic_simd_type<S> && requires(S src, typename S::value_type* dst) {
        aligned_store(internal::abi<S>, src, dst);
    };

struct store_t {
    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, typename T::value_type* dst) noexcept = delete;

    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, typename T::value_type* dst) noexcept
    requires requires { store(internal::abi<T>, src, dst); }
    {
        if consteval {
            [&]<size_t... I>(index_sequence<I...>) {
                (..., (dst[I] = dx::extract(src, imm<I>)));
            }(iota_sequence<T>);
        } else {
            store(internal::abi<T>, src, dst);
        }
    }

    template <basic_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t, T src, typename T::value_type* dst) noexcept {
        if consteval {
            operator()(src, dst);
        } else {
            if constexpr (requires {
                              aligned_store(internal::abi<T>, src, dst);
                          }) {
                aligned_store(internal::abi<T>, src, dst);
            } else {
                operator()(src, dst);
            }
        }
    }

    template <simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, typename T::value_type* dst) noexcept {
        if constexpr (requires { store(internal::abi<T>, src, dst); }) {
            store(internal::abi<T>, src, dst);
        } else if constexpr (basic_simd_element<typename T::value_type>) {
            operator()(dx::to_basic_type(src), dst);
        } else {
            array_for<basic_type_t<T>> array{};
            operator()(dx::to_basic_type(src), array.data);
            __DPL_MEMCPY(dst, array.data, sizeof(array));
        }
    }

    template <simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t tag, T src, typename T::value_type* dst) noexcept {
        if constexpr (requires { aligned_store(internal::abi<T>, src, dst); }) {
            aligned_store(internal::abi<T>, src, dst);
        } else if constexpr (basic_simd_element<typename T::value_type>) {
            operator()(tag, dx::to_basic_type(src), dst);
        } else {
            array_for<basic_type_t<T>> array{};
            operator()(tag, dx::to_basic_type(src), array.data);
            __DPL_MEMCPY(dst, array.data, sizeof(array));
        }
    }
};

struct aligned_store_t : private store_t {
    template <simd_type T>
    requires invocable<store_t, aligned_t, T, typename T::value_type*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t tag, T src, typename T::value_type* dst) noexcept {
        store_t::operator()(tag, src, dst);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::store_t store{};
inline constexpr internal::aligned_store_t aligned_store{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
