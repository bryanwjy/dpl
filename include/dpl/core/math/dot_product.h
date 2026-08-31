// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic/add.h"
#  include "dpl/core/operations/arithmetic/multiply.h"
#  include "dpl/core/operations/internal/reduction.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void dot_product(...) noexcept = delete;

struct dot_product_t :
    public math_operation_base<dot_product_t>,
    public maskable_transform_base<dot_product_t> {
    using operation_base<dot_product_t>::operator();
    using maskable_transform_base<dot_product_t>::operator();
};

template <>
struct operation_signature<dot_product_t> {
    template <simd_vector L, simd_vector R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <typename L, typename R>
concept unqualified_canonical_dot_product = requires {
    {
        dot_product(internal::abi<common_abi_t<L, R>>, internal::declarg<L>(),
            internal::declarg<R>())
    } -> same_as<common_canonical_simd_t<L, R>>;
};

template <typename M, typename L, typename R>
concept unqualified_canonical_mdot_product = requires {
    {
        dot_product(internal::abi<common_abi_t<L, R>>,
            internal::declarg<common_canonical_simd_t<L, R>>(),
            internal::declarg<M>(), internal::declarg<L>(),
            internal::declarg<R>())
    } -> same_as<common_canonical_simd_t<L, R>>;
};

template <typename M, typename L, typename R>
concept unqualified_canonical_mzdot_product = requires {
    {
        dot_product(internal::abi<common_abi_t<L, R>>, dx::zero,
            internal::declarg<M>(), internal::declarg<L>(),
            internal::declarg<R>())
    } -> same_as<common_canonical_simd_t<L, R>>;
};

template <>
struct canonical_impl<dot_product_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = common_canonical_simd_t<L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_dot_product<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(L lhs, R rhs) noexcept {
        return dot_product(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mdot_product<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return dot_product(
            internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<result_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mdot_product<launder_cmask_t<result_t<L, R>, M>,
            L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M cmask, L lhs, R rhs) noexcept {
        return dot_product(internal::abi<common_abi_t<L, R>>, src,
            dx::to_const_mask<result_t<L, R>>(cmask), lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mzdot_product<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return dot_product(
            internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        const_mask_for<result_t<L, R>> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mzdot_product<launder_cmask_t<result_t<L, R>, M>,
            L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return dot_product(internal::abi<common_abi_t<L, R>>, zero,
            dx::to_const_mask<result_t<L, R>>(cmask), lhs, rhs);
    }
};

template <typename L, typename R>
concept unqualified_extended_dot_product = requires {
    {
        dot_product(internal::declarg<L>(), internal::declarg<R>())
    } -> vector_with_common_abi<common_abi_t<L, R>>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mdot_product =
    equivalent_vector_with<S, cpo_result_t<dot_product_t, L, R>> && requires {
        {
            dot_product(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<dot_product_t, L, R>>;
    };

template <typename M, typename L, typename R>
concept unqualified_extended_mzdot_product = requires {
    {
        dot_product(dx::zero, internal::declarg<M>(), internal::declarg<L>(),
            internal::declarg<R>())
    } -> equivalent_vector_with<cpo_result_t<dot_product_t, L, R>>;
};

template <>
struct extended_impl<dot_product_t> {
public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_dot_product<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return dot_product( __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mdot_product<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return dot_product(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mdot_product<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, L&& lhs, R&& rhs) {
        return dot_product(src, dx::to_const_mask<S>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        result_mask_for<dot_product_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mzdot_product<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return dot_product(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, common_vector_with<L> R,
        result_cmask_for<dot_product_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mzdot_product<
            launder_cmask_t<cpo_result_t<dot_product_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs) {
        return dot_product(zero,
            dx::to_const_mask<cpo_result_t<add_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

struct hsum_t;
namespace fwd {
struct fhsum : operation_base<hsum_t> {};
inline constexpr fhsum hsum{};
} // namespace fwd

template <>
struct fallback_impl<dot_product_t> {
    template <canonical_vector L, common_vector_with<L> R>
    requires cpo_invocable<hsum_t, common_canonical_simd_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        L lhs, R rhs) noexcept {
        if constexpr (dx::simd_canonical_invocable<hsum_t,
                          common_canonical_simd_t<L, R>>) {
            // hsum is forward declared here, please include algorithm/hsum
            return fwd::hsum(dx::multiply(lhs, rhs));
        } else {
            return internal::reduction(dx::multiply(lhs, rhs), dx::add);
        }
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::dot_product_t dot_product{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
