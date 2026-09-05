// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep
#include "dpl/core/operations/arithmetic/add.h"
#include "dpl/core/operations/arithmetic/negate.h"
#include "dpl/core/operations/bitwise/bwand.h"

#if !DPL_MODULES
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/const_mask.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/simd_mask_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
/**
 * @brief Fused SIMD sub-add operation using a fixed lane-parity sign mask.
 *
 * Computes:
 *
 * CODE_BLOCK_BEGIN
 * result[i] = a[i] + ((i % 2 == 0) ? +b[i] : -b[i])
 * CODE_BLOCK_END
 *
 * This is the inverse lane-parity convention of @c subadd.
 *
 * @see subadd
 */
void subadd(...) noexcept = delete;

struct DPL_EMPTY_BASES subadd_t :
    public arithmetic_base<subadd_t>,
    public maskable_transform_base<subadd_t> {
    using operation_base<subadd_t>::operator();
    using maskable_transform_base<subadd_t>::operator();
};

template <>
struct operation_signature<subadd_t> {
    template <typename L, typename R>
    requires simd_vector<L> || simd_vector<R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<subadd_t> : binary_canonical_broadcaster<subadd_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<addsub_t, L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

    template <simd_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto make_opmask() noexcept {
        if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
            return []<size_t... Is>(index_sequence<Is...>) {
                return cmask_v<__DPL bitset<sizeof...(Is)>(
                    ((Is & 1) == 0)...)>;
            }(iota_sequence<T>);
        } else {
            auto const idx = dx::lane_index<signed_canonical_vector_t<T>>();
            return dx::cmpeq(dx::bwand(idx, dx::one), dx::zero);
        }
    }

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && cpo_invocable<add_t, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr common_canonical_simd_t<L, R>
        DPL_VECTORCALL operator()(L lhs, R rhs) noexcept {
        auto const opmask = make_opmask<R>();
        return dx::add(lhs, dx::negate(rhs, opmask, rhs));
    }

    using binary_canonical_broadcaster<subadd_t>::operator();

    template <unextended_type L, unextended_terminal_of<subadd_t, L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr result_t<L, R>
        DPL_VECTORCALL operator()(
            result_t<L, R> src, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        using DL DPL_NODEBUG = decay_t<L>;
        using DR DPL_NODEBUG = decay_t<R>;
        if constexpr (broadcastable_to<L, DR>) {
            return addsub_t::operator()(src, mask,
                dx::broadcast<DR>(__DPL forward<L>(lhs)),
                __DPL forward<R>(rhs));
        } else if constexpr (broadcastable_to<R, DL>) {
            return addsub_t::operator()(src, mask, __DPL forward<L>(lhs),
                dx::broadcast<DL>(__DPL forward<R>(rhs)));
        } else {
            auto const opmask = make_opmask<R>();
            return dx::add(src, mask, lhs, dx::negate(rhs, opmask, rhs));
        }
    }

    template <unextended_type L, unextended_type R,
        result_cmask_for<addsub_t, L, R> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr result_t<L, R>
        DPL_VECTORCALL operator()(
            result_t<L, R> src, M mask, L&& lhs, R&& rhs) noexcept {
        using DL DPL_NODEBUG = decay_t<L>;
        using DR DPL_NODEBUG = decay_t<R>;
        if constexpr (broadcastable_to<L, DR>) {
            return addsub_t::operator()(src, mask,
                dx::broadcast<DR>(__DPL forward<L>(lhs)),
                __DPL forward<R>(rhs));
        } else if constexpr (broadcastable_to<R, DL>) {
            return addsub_t::operator()(src, mask, __DPL forward<L>(lhs),
                dx::broadcast<DL>(__DPL forward<R>(rhs)));
        } else {
            auto const opmask = make_opmask<R>();
            return dx::add(src, mask, lhs, dx::negate(rhs, opmask, rhs));
        }
    }

    template <unextended_type L, unextended_terminal_of<subadd_t, L> R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr result_t<L, R>
        DPL_VECTORCALL operator()(
            dx::zero_t zero, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        using DL DPL_NODEBUG = decay_t<L>;
        using DR DPL_NODEBUG = decay_t<R>;
        if constexpr (broadcastable_to<L, DR>) {
            return addsub_t::operator()(zero, mask,
                dx::broadcast<DR>(__DPL forward<L>(lhs)),
                __DPL forward<R>(rhs));
        } else if constexpr (broadcastable_to<R, DL>) {
            return addsub_t::operator()(zero, mask, __DPL forward<L>(lhs),
                dx::broadcast<DL>(__DPL forward<R>(rhs)));
        } else {
            auto const opmask = make_opmask<R>();
            return dx::add(zero, mask, lhs, dx::negate(rhs, opmask, rhs));
        }
    }

    template <unextended_type L, unextended_type R,
        result_cmask_for<addsub_t, L, R> M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr result_t<L, R>
        DPL_VECTORCALL operator()(
            dx::zero_t zero, M mask, L&& lhs, R&& rhs) noexcept {
        using DL DPL_NODEBUG = decay_t<L>;
        using DR DPL_NODEBUG = decay_t<R>;
        if constexpr (broadcastable_to<L, DR>) {
            return addsub_t::operator()(zero, mask,
                dx::broadcast<DR>(__DPL forward<L>(lhs)),
                __DPL forward<R>(rhs));
        } else if constexpr (broadcastable_to<R, DL>) {
            return addsub_t::operator()(zero, mask, __DPL forward<L>(lhs),
                dx::broadcast<DL>(__DPL forward<R>(rhs)));
        } else {
            auto const opmask = make_opmask<R>();
            return dx::add(zero, mask, lhs, dx::negate(rhs, opmask, rhs));
        }
    }
};

template <typename L, typename R, typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_subadd = requires {
    {
        subadd(internal::abi<T>, internal::declarg<L>(), internal::declarg<R>())
    } -> same_as<T>;
};

template <typename S, typename M, typename L, typename R, typename T>
concept unqualified_canonical_msubadd_base =
    cpo_invocable<subadd_t, L, R> && requires {
        {
            subadd(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<T>;
    };

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_msubadd =
    unqualified_canonical_msubadd_base<T, M, L, R, T>;

template <typename M, typename L, typename R,
    typename T = common_canonical_simd_t<L, R>>
concept unqualified_canonical_zmsubadd =
    unqualified_canonical_msubadd_base<dx::zero_t, M, L, R, T>;

template <>
struct canonical_impl<subadd_t> {
private:
    template <typename L, typename R>
    using result_t DPL_NODEBUG = cpo_result_t<subadd_t, L, R>;
    template <typename L, typename R>
    using mask_t DPL_NODEBUG = simd_mask_type_t<result_t<L, R>>;

public:
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R> && unqualified_canonical_subadd<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        L lhs, R rhs) noexcept {
        return subadd(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R>
    requires unqualified_canonical_subadd<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R&& rhs) noexcept {
        return subadd(internal::abi<L>, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L>
    requires unqualified_canonical_subadd<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr R operator()(L&& lhs, R rhs) noexcept {
        return subadd(internal::abi<R>, __DPL forward<L>(lhs), rhs);
    }

    template <unextended_type L, unextended_terminal_of<subadd_t, L> R>
    requires unqualified_canonical_msubadd<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return subadd(internal::abi<result_t<L, R>>, src, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_msubadd<launder_cmask_t<result_t<L, R>, M>,
        L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        result_t<L, R> src, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return subadd(internal::abi<T>, src, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_terminal_of<subadd_t, L> R>
    requires unqualified_canonical_zmsubadd<mask_t<L, R>, L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, mask_t<L, R> mask, L&& lhs, R&& rhs) noexcept {
        return subadd(internal::abi<result_t<L, R>>, zero, mask,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <unextended_type L, unextended_type R, result_cmask_for<L, R> M>
    requires unqualified_canonical_zmsubadd<launder_cmask_t<result_t<L, R>, M>,
        L, R, result_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<L, R> operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) noexcept {
        using T = result_t<L, R>;
        return subadd(internal::abi<T>, zero, dx::to_const_mask<T>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <typename L, typename R, typename T>
concept unqualified_extended_subadd = requires {
    {
        subadd(internal::declarg<L>(), internal::declarg<R>())
    } -> equivalent_vector_with<T>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_msubadd_base =
    cpo_invocable<subadd_t, L, R> && requires {
        {
            subadd(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<cpo_result_t<subadd_t, L, R>>;
    };

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_msubadd =
    equivalent_vector_with<S, cpo_result_t<subadd_t, L, R>> &&
    unqualified_extended_msubadd_base<S, M, L, R>;

template <typename M, typename L, typename R>
concept unqualified_extended_zmsubadd =
    unqualified_extended_msubadd_base<dx::zero_t, M, L, R>;

template <>
struct extended_impl<subadd_t> {
public:
    template <simd_vector L, common_vector_with<L> R>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_subadd<L, R, common_canonical_simd_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return subadd(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector L, broadcastable_to<L> R>
    requires unqualified_extended_subadd<L, R, L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return subadd(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <extended_vector R, broadcastable_to<R> L>
    requires unqualified_extended_subadd<L, R, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return subadd(__DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_msubadd<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return subadd(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msubadd<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, L&& lhs, R&& rhs) {
        return subadd(src, dx::to_const_mask<S>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_mask_for<subadd_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmsubadd<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return subadd(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <typename L, typename R, result_cmask_for<subadd_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_zmsubadd<
            launder_cmask_t<cpo_result_t<subadd_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, L&& lhs, R&& rhs) {
        return subadd(zero,
            dx::to_const_mask<cpo_result_t<subadd_t, L, R>>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::subadd_t subadd{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
