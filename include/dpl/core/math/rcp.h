// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/fma.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/utility/fpfix.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rcp(...) noexcept = delete;

struct DPL_EMPTY_BASES rcp_t :
    private math_operation_base<rcp_t>,
    private maskable_transform_base<rcp_t> {
    using math_operation_base<rcp_t>::operator();
    using maskable_transform_base<rcp_t>::operator();
};

template <>
struct operation_signature<rcp_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mrcp =
    (!simd_type<S> || same_as<S, cpo_result_t<rcp_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            rcp(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<rcp_t, T>>;
    };

template <>
struct canonical_impl<rcp_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> M>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, M>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { rcp(internal::abi<A>, val); }
    {
        return rcp(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrcp<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return rcp(internal::abi<T>, src, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mrcp<type_identity_t<T>, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, cmask_t<T, M> cmask, T val) noexcept {
        return rcp(internal::abi<T>, src, cmask, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrcp<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return rcp(internal::abi<T>, zero, mask, val);
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mrcp<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T val) noexcept {
        return rcp(internal::abi<T>, zero, cmask, val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_rcp = requires {
    { rcp(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mrcp =
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<rcp_t, T>>) &&
    requires {
        {
            rcp(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<rcp_t, T>>;
    };

template <>
struct extended_impl<rcp_t> {
private:
    template <typename T>
    using imask_t DPL_NODEBUG = mask_value_t<simd_abi_traits<T>::size>;

    template <typename T, imask_t<T> M>
    using cmask_t DPL_NODEBUG = const_mask<simd_abi_traits<T>::size, M>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <extended_vector T>
    requires unqualified_extended_rcp<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return rcp(__DPL forward<T>(val));
    }

    template <simd_vector S, common_vector_with<S> T,
        equivalent_mask_with<mask_t<S>> M>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrcp<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return rcp( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <fixed_width_vector S, imask_t<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrcp<S, cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, cmask_t<S, M> cmask, T&& val) {
        return rcp( __DPL forward<S>(src), cmask, __DPL forward<T>(val));
    }

    template <simd_vector T, common_mask_with<mask_t<T>> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrcp<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return rcp(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <fixed_width_vector T, imask_t<T> M>
    requires extended_vector<T> &&
        unqualified_extended_mrcp<dx::zero_t, cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, cmask_t<T, M> cmask, T&& val) {
        return rcp(zero, cmask, __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<rcp_t> {
private:
    template <typename E>
    static constexpr auto useed = []() {
        if constexpr (same_as<E, double>) {
            return unsigned_representation_t<E>(0x7FDE6238DA3C2118);
        } else if constexpr (same_as<E, float>) {
            return unsigned_representation_t<E>(0x7EF311C3);
        } else if constexpr (brain_float<E>) {
            return unsigned_representation_t<E>(0x7EF3);
        } else {
            static_assert(sizeof(E) == 2);
            return unsigned_representation_t<E>(0x7800);
        }
    }();

    template <typename E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto refine(
        basic_vector<E, A> y, basic_vector<E, A> x) noexcept {
        return y * dx::fnmadd(x, y, 2.0);
    }

    template <typename E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<E, A> approximate(
        basic_vector<E, A> val) noexcept {
        auto const seed =
            useed<E> - dx::reinterpret<unsigned_representation_t<E>>(val);
        return refine(dx::reinterpret<E>(seed), val);
    }

    static constexpr auto fix = fpfix::condition<fpfix::infinity, dx::zero> |
        fpfix::condition<fpfix::zero, fpfix::signed_inf>;

public:
    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<float, A> val) noexcept {
        // nan is implicitly handled
        return dx::fixup(val, approximate(val), fix);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<double, A> val) noexcept {
        // nan is implicitly handled
        return dx::fixup(val, approximate(val), fix);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::rcp_t rcp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
