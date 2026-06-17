// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/aligned.h"
#include "dpl/core/basic/internal/abi.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/base.h"
#  include "dpl/core/dispatch/operation/basic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void store(...) noexcept = delete;

struct store_t :
    private basic_operation_base<store_t>,
    private maskable_operation_base<store_t> {
    using operation_base<store_t>::operator();
};

template <>
struct operation_signature<store_t> {
    template <canonical_simd_type T>
    static consteval void operator()(T&&, simd_element_type_t<T>*) noexcept {}
    template <canonical_simd_type T>
    static consteval void operator()(
        aligned_t, T&&, simd_element_type_t<T>*) noexcept {}
    // TODO maskable signatures
};

template <>
struct canonical_impl<store_t> {
    // TODO maskable signatures
    template <canonical_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, simd_element_type_t<T>* ptr) noexcept
    requires requires { store(internal::abi<T>, src, ptr); }
    {
        return store(internal::abi<T>, src, ptr);
    }

    template <canonical_simd_type T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t aligned, T src, simd_element_type_t<T>* ptr) noexcept
    requires requires { store(internal::abi<T>, aligned, src, ptr); }
    {
        return store(internal::abi<T>, aligned, src, ptr);
    }
};

template <>
struct fallback_impl<store_t> {
    template <canonical_simd_type T>
    requires cpo_invocable<canonical_impl<store_t>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t aligned, T src, simd_element_type_t<T>* ptr) noexcept {
        return canonical_impl<store_t>::operator()(src, ptr);
    }
};

struct aligned_store_t {
    template <canonical_simd_type T>
    requires cpo_invocable<store_t, aligned_t, T, simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, simd_element_type_t<T>* dst) noexcept {
        store_t::operator()(dx::aligned, __DPL forward<T>(src), dst);
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
