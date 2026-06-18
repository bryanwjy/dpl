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
#  include "dpl/core/immediate/const_mask.h"
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
    template <canonical_vector T>
    static consteval void operator()(T&&, simd_element_type_t<T>*) noexcept {}
    template <canonical_vector T>
    static consteval void operator()(
        aligned_t, T&&, simd_element_type_t<T>*) noexcept {}
};

template <typename A, typename... Ts>
concept unqualified_store =
    requires { store(internal::abi<A>, internal::declarg<Ts>()...); };

template <>
struct canonical_impl<store_t> {
private:
    template <simd_vector T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <canonical_vector T>
    requires unqualified_store<simd_abi_type_t<T>, T, simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, simd_element_type_t<T>* ptr) noexcept {
        return store(internal::abi<T>, src, ptr);
    }

    template <canonical_vector T>
    requires unqualified_store<simd_abi_type_t<T>, mask_t<T>, T,
        simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        mask_t<T> mask, T src, simd_element_type_t<T>* ptr) noexcept {
        return store(internal::abi<T>, mask, src, ptr);
    }

    template <fixed_width_vector T, const_mask_for<T> M>
    requires canonical_vector<T> &&
        unqualified_store<simd_abi_type_t<T>, launder_cmask_t<T, M>, T,
            simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        M mask, T src, simd_element_type_t<T>* ptr) noexcept {
        constexpr auto cmask = dx::to_const_mask<T>(mask);
        return store(internal::abi<T>, cmask, src, ptr);
    }

    template <canonical_vector T>
    requires unqualified_store<simd_abi_type_t<T>, aligned_t, T,
        simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        aligned_t aligned, T src, simd_element_type_t<T>* ptr) noexcept {
        return store(internal::abi<T>, aligned, src, ptr);
    }

    template <canonical_vector T>
    requires unqualified_store<simd_abi_type_t<T>, mask_t<T>, aligned_t, T,
        simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(mask_t<T> mask, aligned_t aligned, T src,
        simd_element_type_t<T>* ptr) noexcept {
        return store(internal::abi<T>, mask, aligned, src, ptr);
    }

    template <fixed_width_vector T, const_mask_for<T> M>
    requires canonical_vector<T> &&
        unqualified_store<simd_abi_type_t<T>, launder_cmask_t<T, M>, aligned_t,
            T, simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(M mask, aligned_t aligned, T src,
        simd_element_type_t<T>* ptr) noexcept {
        constexpr auto cmask = dx::to_const_mask<T>(mask);
        return store(internal::abi<T>, cmask, aligned, src, ptr);
    }
};

struct aligned_store_t {
    template <canonical_vector T>
    requires cpo_invocable<store_t, aligned_t, T, simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        T src, simd_element_type_t<T>* dst) noexcept {
        store_t::operator()(dx::aligned, __DPL forward<T>(src), dst);
    }

    template <typename M, canonical_vector T>
    requires cpo_invocable<store_t, M, aligned_t, T, simd_element_type_t<T>*>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE)
    static constexpr void operator()(
        M mask, T src, simd_element_type_t<T>* dst) noexcept {
        store_t::operator()(mask, dx::aligned, __DPL forward<T>(src), dst);
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
