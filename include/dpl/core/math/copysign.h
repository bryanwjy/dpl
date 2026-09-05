// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/signbit.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise/bwxor.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void copysign(...) noexcept = delete;

struct DPL_EMPTY_BASES copysign_t :
    public math_operation_base<copysign_t>,
    public maskable_transform_base<copysign_t> {
    using math_operation_base<copysign_t>::operator();
    using maskable_transform_base<copysign_t>::operator();
};

template <>
struct operation_signature<copysign_t> {
    template <typename L, simd_vector R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<copysign_t> {
    template <canonical_vector L, vector_subsumed_by<L> R>
    requires canonical_vector<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr L DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        using E = simd_element_type_t<L>;
        if constexpr (unsigned_integral<E>) {
            return lhs;
        } else if constexpr (floating_point_like<E>) {
            constexpr auto signbit =
                __DPL bit_cast<E>(floating_point_traits<E>::signbit);
            return dx::bwxor(lhs, dx::bwand(dx::bwxor(lhs, rhs), signbit));
        } else {
            return dx::negate(lhs, dx::signbit(dx::bwxor(lhs, rhs)), lhs);
        }
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires cpo_invocable<copysign_t, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr R DPL_VECTORCALL operator()(L&& lhs, R rhs) noexcept {
        return copysign_t::operator()(
            dx::broadcast<R>(__DPL forward<L>(lhs)), rhs);
    }
};

template <typename L, typename R, typename T>
concept unqualified_canonical_copysign = requires {
    {
        copysign(
            internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<T>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_mcopysign_base =
    cpo_invocable<copysign_t, L, R> && requires {
        {
            copysign(internal::abi<cpo_result_t<copysign_t, L, R>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> same_as<cpo_result_t<copysign_t, L, R>>;
    };

template <typename M, typename L, typename R>
concept unqualified_canonical_mcopysign = cpo_invocable<copysign_t, L, R> &&
    unqualified_canonical_mcopysign_base<cpo_result_t<copysign_t, L, R>, M, L,
        R>;

template <typename M, typename L, typename R>
concept unqualified_canonical_zmcopysign =
    unqualified_canonical_mcopysign_base<dx::zero_t, M, L, R>;

template <>
struct canonical_impl<copysign_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<copysign_t, L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, vector_subsumed_by<L> R>
    requires canonical_vector<R> && unqualified_canonical_copysign<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R rhs) noexcept {
        return copysign(internal::abi<L>, lhs, rhs);
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_copysign<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs) noexcept {
        return copysign(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <unextended_type L, unextended_terminal_of<copysign_t, L> R>
    requires unqualified_canonical_mcopysign<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return copysign(internal::abi<result_t<L, R>>, src, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R,
        result_cmask_for<copysign_t, L, R> M>
    requires unqualified_canonical_mcopysign<launder_cmask_t<result_t<L, R>, M>,
        L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M mask, L&& lhs, R&& rhs) noexcept {
        return copysign(internal::abi<result_t<L, R>>, src,
            dx::to_const_mask<result_t<L, R>>(mask),__DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_terminal_of<copysign_t, L> R>
    requires unqualified_canonical_zmcopysign<mask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return copysign(internal::abi<result_t<L, R>>, zero, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R,
        result_cmask_for<copysign_t, L, R> M>
    requires unqualified_canonical_zmcopysign<
        launder_cmask_t<result_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) noexcept {
        return copysign(internal::abi<result_t<L, R>>, zero,
            dx::to_const_mask<result_t<L, R>>(mask),__DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }
};

template <typename L, typename R, typename T>
concept unqualified_extended_copysign = requires {
    {
        copysign(internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mcopysign_base =
    cpo_invocable<copysign_t, L, R> && requires {
        {
            copysign(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<copysign_t, L, R>>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_mcopysign =
    unqualified_extended_mcopysign_base<S, M, L, R> &&
    equivalent_vector_with<S, cpo_result_t<copysign_t, L, R>>;

template <typename M, typename L, typename R>
concept unqualified_extended_zmcopysign =
    unqualified_extended_mcopysign_base<dx::zero_t, M, L, R>;

template <>
struct extended_impl<copysign_t> {
public:
    template <simd_vector L, vector_subsumed_by<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_copysign<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return copysign(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_copysign<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return copysign(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mcopysign<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return copysign(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mcopysign<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, L&& lhs, R&& rhs) {
        return copysign(__DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_mask_for<copysign_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmcopysign<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return copysign(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_cmask_for<copysign_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmcopysign<
            launder_cmask_t<cpo_result_t<copysign_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) {
        return copysign(zero,
            dx::to_const_mask<cpo_result_t<copysign_t, L, R>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::copysign_t copysign{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
