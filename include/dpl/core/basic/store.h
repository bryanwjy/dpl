// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/aligned.h"
#include "dpl/core/basic/internal/abi.h"
#include "dpl/core/basic/to_canonical.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/canonical.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void store(...) noexcept = delete;
void aligned_store(...) noexcept = delete;

struct store_t {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, simd_element_type_t<T>* dst) noexcept
    requires requires { store(internal::abi<T>, src, dst); }
    {
        store(internal::abi<T>, src, dst);
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t, T src, simd_element_type_t<T>* dst) noexcept {
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

    template <simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, simd_element_type_t<T>* dst) noexcept {
        if constexpr (requires { store(src, dst); }) {
            store(src, dst);
        } else {
            static_assert(is_trivially_copyable_v<simd_element_type_t<T>>);
            operator()(dx::to_canonical(src), dst);
        }
    }

    template <simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t tag, T src, simd_element_type_t<T>* dst) noexcept {
        if constexpr (requires { aligned_store(src, dst); }) {
            aligned_store(src, dst);
        } else {
            static_assert(is_trivially_copyable_v<simd_element_type_t<T>>);
            operator()(tag, dx::to_canonical(src), dst);
        }
    }
};

struct aligned_store_t : private store_t {
    template <simd_vector T>
    requires invocable<store_t, aligned_t, T, simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t tag, T src, simd_element_type_t<T>* dst) noexcept {
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
