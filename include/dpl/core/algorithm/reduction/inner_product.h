// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduction/hsum.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/arithmetic/add.h"
#  include "dpl/core/operations/arithmetic/multiply.h"
#  include "dpl/core/operations/cast/element_cast.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void inner_product(...) noexcept = delete;

struct inner_product_t :
    private exclusive_scan_base<inner_product_t>,
    private maskable_accumulation_base<inner_product_t> {
    using operation_base<inner_product_t>::operator();
    using maskable_accumulation_base<inner_product_t>::operator();
};

template <>
struct operation_signature<inner_product_t> {
    template <simd_vector S, simd_vector L, simd_vector R>
    static consteval void operator()(S&&, L&&, R&&) noexcept {}
    template <simd_vector L, simd_vector R>
    static consteval void operator()(dx::zero_t, L&&, R&&) noexcept {}
};

template <typename S, typename L, typename R>
concept unqualified_canonical_inner_product =
    same_as<basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>, S> &&
    requires {
        {
            inner_product(internal::abi<common_abi_t<L, R>>,
                internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<S>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_minner_product =
    same_as<S, cpo_result_t<inner_product_t, S, L, R>> && requires {
        {
            inner_product(internal::abi<cpo_result_t<inner_product_t, S, L, R>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<inner_product_t, S, L, R>>;
    };

template <typename M, typename S, typename L, typename R>
concept unqualified_canonical_mzinner_product =
    cpo_invocable<inner_product_t, S, L, R> && requires {
        {
            inner_product(internal::abi<cpo_result_t<inner_product_t, S, L, R>>,
                dx::zero, internal::declarg<M>(), internal::declarg<S>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<inner_product_t, S, L, R>>;
    };

template <>
struct canonical_impl<inner_product_t> {
public:
    template <typename L, typename R>
    using source_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using imask_t DPL_NODEBUG = mask_value_t<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size>;

    template <typename L, typename R, imask_t<L, R> V>
    using cmask_t DPL_NODEBUG = const_mask<
        simd_abi_traits<simd_element_type_t<L>, common_abi_t<L, R>>::size, V>;

private:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_inner_product<source_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr source_t<L, R> operator()(
        source_t<L, R> src, L lhs, R rhs) noexcept {
        return inner_product(internal::abi<common_abi_t<L, R>>, src, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R,
        simd_element_for<common_abi_t<L, R>> SE>
    requires canonical_vector<R> &&
        unqualified_canonical_minner_product<source_t<L, R>, mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr source_t<L, R> operator()(
        source_t<L, R> src, mask_t<L, R> mask, L lhs, R rhs) noexcept {
        return inner_product(
            internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_minner_product<source_t<L, R>, cmask_t<L, R, M>,
            L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr source_t<L, R> operator()(
        source_t<L, R> src, cmask_t<L, R, M> cmask, L lhs, R rhs) noexcept {
        return inner_product(
            internal::abi<common_abi_t<L, R>>, src, cmask, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> &&
        unqualified_canonical_mzinner_product<mask_t<L, R>, source_t<L, R>, L,
            R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, mask_t<L, R> mask,
        source_t<L, R> src, L lhs, R rhs) noexcept {
        return inner_product(
            internal::abi<common_abi_t<L, R>>, zero, mask, src, lhs, rhs);
    }

    template <canonical_vector L, common_vector_with<L> R, imask_t<L, R> M>
    requires canonical_vector<R> &&
        unqualified_canonical_mzinner_product<cmask_t<L, R, M>, source_t<L, R>,
            L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, cmask_t<L, R, M> cmask,
        source_t<L, R> src, L lhs, R rhs) noexcept {
        return inner_product(
            internal::abi<common_abi_t<L, R>>, zero, cmask, src, lhs, rhs);
    }
};

template <typename S, typename L, typename R>
concept unqualified_extended_inner_product =
    same_as<simd_abi_type_t<S>, common_abi_t<L, R>> && requires {
        {
            inner_product(internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_vector_with<S>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_minner_product =
    equivalent_vector_with<S, cpo_result_t<inner_product_t, S, L, R>> &&
    requires {
        {
            inner_product(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<inner_product_t, S, L, R>>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mzinner_product =
    equivalent_vector_with<S, cpo_result_t<inner_product_t, S, L, R>> &&
    requires {
        {
            inner_product(dx::zero, internal::declarg<M>(),
                internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<inner_product_t, S, L, R>>;
    };

template <>
struct extended_impl<inner_product_t> {
private:
    template <typename S>
    using simask_t DPL_NODEBUG = mask_value_t<simd_abi_type_t<S>::size>;

    template <typename S, simask_t<S> V>
    using scmask_t DPL_NODEBUG = const_mask<simd_abi_type_t<S>::size, V>;

public:
    template <simd_vector S, simd_vector L, common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_inner_product<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return inner_product(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_minner_product<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return inner_product(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, simask_t<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_minner_product<S, scmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        S&& src, scmask_t<S, M> cmask, L&& lhs, R&& rhs) {
        return inner_product( __DPL forward<S>(src), cmask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R> ||
                 extended_mask<M>) &&
        unqualified_extended_mzinner_product<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, S&& src, L&& lhs, R&& rhs) {
        return inner_product(zero, __DPL forward<M>(mask),
            __DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector S, simask_t<S> M, simd_vector L,
        common_vector_with<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mzinner_product<S, scmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, scmask_t<S, M> cmask, S&& src, L&& lhs, R&& rhs) {
        return inner_product(zero, cmask, __DPL forward<S>(src),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <>
struct fallback_impl<inner_product_t> {
    template <simd_vector S, simd_vector L, simd_vector R>
    requires cpo_invocable<multiply_t, L, R> &&
        cpo_invocable<hsum_t, cpo_result_t<multiply_t, L, R>> &&
        cpo_invocable<add_t, S,
            cpo_result_t<hsum_t, cpo_result_t<multiply_t, L, R>>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return dx::add(__DPL forward<S>(src),
            dx::hsum(
                dx::multiply(__DPL forward<L>(lhs), __DPL forward<R>(rhs))));
    }

    template <simd_vector L, simd_vector R>
    requires cpo_invocable<multiply_t, L, R> &&
        cpo_invocable<hsum_t, cpo_result_t<multiply_t, L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, L&& lhs, R&& rhs) {
        return dx::hsum(
            dx::multiply(__DPL forward<L>(lhs), __DPL forward<R>(rhs)));
    }

    template <simd_vector S, typename M, simd_vector L, simd_vector R>
    requires cpo_invocable<inner_product_t, S, L, R> &&
        cpo_invocable<select_t, M, cpo_result_t<inner_product_t, S, L, R>,
            S const&>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        S&& src, M&& mask, L&& lhs, R&& rhs) {
        auto const osrc = src;
        return dx::select(__DPL forward<M>(mask),
            inner_product_t::operator()(__DPL forward<S>(src),
                __DPL forward<L>(lhs), __DPL forward<R>(rhs)),
            osrc);
    }

    template <typename M, simd_vector S, simd_vector L, simd_vector R>
    requires cpo_invocable<inner_product_t, S, L, R> &&
        cpo_invocable<select_t, M, cpo_result_t<inner_product_t, S, L, R>,
            dx::zero_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        dx::zero_t zero, M&& mask, S&& src, L&& lhs, R&& rhs) {
        return dx::select(__DPL forward<M>(mask),
            inner_product_t::operator()(__DPL forward<S>(src),
                __DPL forward<L>(lhs), __DPL forward<R>(rhs)),
            zero);
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::inner_product_t inner_product{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
