// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/fpfix.h"
#include "dpl/core/math/mulx.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/numbers/ext.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/type_traits/representation.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rcp(...) noexcept = delete;

struct DPL_EMPTY_BASES rcp_t :
    public math_operation_base<rcp_t>,
    public maskable_transform_base<rcp_t> {
    using math_operation_base<rcp_t>::operator();
    using maskable_transform_base<rcp_t>::operator();
};

template <>
struct operation_signature<rcp_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mrcp = cpo_invocable<rcp_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<rcp_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            rcp(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<rcp_t, T>>;
    };

template <>
struct canonical_impl<rcp_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { rcp(internal::abi<T>, val); }
    {
        return rcp(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrcp<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return rcp(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrcp<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return rcp(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrcp<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return rcp(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrcp<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return rcp(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_rcp = requires {
    { rcp(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mrcp = cpo_invocable<rcp_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<rcp_t, T>>) &&
    requires {
        {
            rcp(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<rcp_t, T>>;
    };

template <>
struct extended_impl<rcp_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_rcp<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return rcp(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrcp<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return rcp( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrcp<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return rcp( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<rcp_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrcp<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return rcp(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<rcp_t, T> M>
    requires unqualified_extended_mrcp<dx::zero_t,
        launder_cmask_t<cpo_result_t<rcp_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return rcp(zero, dx::to_const_mask<cpo_result_t<rcp_t, T>>(cmask),
            __DPL forward<T>(val));
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
        } else if constexpr (same_as<E, ext::bfloat16>) {
            return unsigned_representation_t<E>(0x7EF3);
        } else {
            static_assert(sizeof(E) == 2);
            return unsigned_representation_t<E>(0x7800);
        }
    }();

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto refine(T y, T x) noexcept {
        return dx::multiply(y, dx::nmuladd(x, y, 2.0));
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T approximate(T val) noexcept {
        using E = simd_element_type_t<T>;
        using uint_t = unsigned_representation_t<E>;
        auto const seed = dx::subtract(useed<E>, dx::reinterpret<uint_t>(val));
        return refine(dx::reinterpret<E>(seed), val);
    }

    static constexpr auto fix = fpfix::condition<fpfix::infinity, dx::zero> |
        fpfix::condition<fpfix::zero, fpfix::signed_inf>;

public:
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        // nan is implicitly handled
        return dx::fixup(val, approximate(val), fix);
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
        // nan is implicitly handled
        return dx::fixup(val, approximate(val), fix);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::rcp_t rcp{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
