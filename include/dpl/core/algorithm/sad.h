// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduction/hsum.h"

#if !DPL_MODULES
#  include "dpl/core/basic/lane_index.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/dispatch/maskable/accumulation.h"
#  include "dpl/core/dispatch/operation/algorithm.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bitwise.h"
#  include "dpl/core/operations/compare.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
void sad(...) noexcept = delete;

/**
 * @brief Sum of Absolute Differences (SAD) between two integer vectors.
 */
struct sad_t :
    public algorithm_base<sad_t>,
    public maskable_accumulation_base<sad_t> {
    using operation_base<sad_t>::operator();
    using maskable_accumulation_base<sad_t>::operator();
};

template <>
struct operation_signature<sad_t> {
    template <simd_vector S, simd_vector L, simd_vector R>
    static consteval void operator()(S&&, L&&, R&&) noexcept {}
    template <simd_vector L, simd_vector R>
    static consteval void operator()(dx::zero_t, L&&, R&&) noexcept {}
};

template <typename S, typename L, typename R>
concept sad_result = simd_vector<L> && simd_vector<R> && simd_vector<S> &&
    same_abi_as<simd_abi_type_t<S>, common_abi_t<L, R>>;

template <typename S, typename L, typename R>
concept canonical_sad_result = canonical_vector<L> && canonical_vector<R> &&
    canonical_vector<S> && sad_result<S, L, R>;

template <typename S, typename L, typename R>
concept unqualified_canonical_sad = requires {
    {
        sad(internal::abi<S>, internal::declarg<S>(), internal::declarg<L>(),
            internal::declarg<R>())
    } -> same_as<S>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_canonical_msad = cpo_invocable<sad_t, S, L, R> &&
    same_as<S, cpo_result_t<sad_t, S, L, R>> && requires {
        {
            sad(internal::abi<S>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<S>;
    };

template <typename M, typename S, typename L, typename R>
concept unqualified_canonical_zmsad = cpo_invocable<sad_t, S, L, R> &&
    same_as<S, cpo_result_t<sad_t, S, L, R>> && requires {
        {
            sad(internal::abi<S>, dx::zero, internal::declarg<M>(),
                internal::declarg<S>(), internal::declarg<L>(),
                internal::declarg<R>())
        } -> same_as<S>;
    };

template <>
struct canonical_impl<sad_t> {
private:
    template <canonical_vector L, common_vector_with<L> R,
        canonical_sad_result<L, R> S>
    requires unqualified_canonical_sad<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L lhs, R rhs) noexcept {
        return sad(internal::abi<S>, src, lhs, rhs);
    }

    template <canonical_vector L, broadcastable_to<L> R,
        canonical_sad_result<L, L> S>
    requires unqualified_canonical_sad<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L lhs, R&& rhs) noexcept {
        return sad(internal::abi<S>, src, lhs, __DPL forward<R>(rhs));
    }

    template <canonical_vector R, broadcastable_to<R> L,
        canonical_sad_result<R, R> S>
    requires unqualified_canonical_sad<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, L&& lhs, R rhs) noexcept {
        return sad(internal::abi<S>, src, __DPL forward<L>(lhs), rhs);
    }

    template <canonical_vector S, unextended_type L, unextended_type R>
    requires unqualified_canonical_msad<S, simd_mask_type_t<S>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(
        S src, simd_mask_type_t<S> mask, L&& lhs, R&& rhs) noexcept {
        return sad(internal::abi<S>, src, mask, __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <canonical_vector S, const_mask_for<S> M, unextended_type L,
        unextended_type R>
    requires unqualified_canonical_msad<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr S operator()(S src, M mask, L&& lhs, R&& rhs) noexcept {
        return sad(internal::abi<S>, src, dx::to_const_mask<S>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <canonical_vector S, unextended_type L, unextended_type R>
    requires unqualified_canonical_zmsad<simd_mask_type_t<S>, S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, simd_mask_type_t<S> mask,
        S src, L&& lhs, R&& rhs) noexcept {
        return sad(internal::abi<S>, zero, mask, src, __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <canonical_vector S, const_mask_for<S> M, unextended_type L,
        unextended_type R>
    requires unqualified_canonical_zmsad<launder_cmask_t<S, M>, S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, S src, L&& lhs, R&& rhs) noexcept {
        return sad(internal::abi<S>, zero, dx::to_const_mask<S>(mask), src,
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <typename S, typename L, typename R>
concept unqualified_extended_sad = requires {
    {
        sad(internal::declarg<S>(), internal::declarg<L>(),
            internal::declarg<R>())
    } -> equivalent_vector_with<S>;
};

template <typename S, typename M, typename L, typename R>
concept unqualified_extended_msad = cpo_invocable<sad_t, S, L, R> &&
    equivalent_vector_with<S, cpo_result_t<sad_t, S, L, R>> && requires {
        {
            sad(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<S>;
    };

template <typename M, typename S, typename L, typename R>
concept unqualified_extended_mzsad = cpo_invocable<sad_t, S, L, R> &&
    equivalent_vector_with<S, cpo_result_t<sad_t, S, L, R>> && requires {
        {
            sad(dx::zero, internal::declarg<M>(), internal::declarg<S>(),
                internal::declarg<L>(), internal::declarg<R>())
        } -> equivalent_vector_with<S>;
    };

template <>
struct extended_impl<sad_t> {
public:
    template <simd_vector L, common_vector_with<L> R, sad_result<L, R> S>
    requires (extended_vector<L> || extended_vector<R> || extended_vector<S>) &&
        unqualified_extended_sad<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return sad(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, broadcastable_to<L> R, sad_result<L, L> S>
    requires (extended_vector<L> || extended_vector<S>) &&
        unqualified_extended_sad<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return sad(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector R, broadcastable_to<R> L, sad_result<R, R> S>
    requires (extended_vector<R> || extended_vector<S>) &&
        unqualified_extended_sad<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) {
        return sad(__DPL forward<S>(src), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_msad<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return sad(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_msad<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, L&& lhs, R&& rhs) {
        return sad( __DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R> ||
                 extended_mask<M>) &&
        unqualified_extended_mzsad<M, S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, S&& src, L&& lhs, R&& rhs) {
        return sad(zero, __DPL forward<M>(mask), __DPL forward<S>(src),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, typename L, typename R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mzsad<launder_cmask_t<S, M>, S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero,
        launder_cmask_t<S, M> mask, S&& src, L&& lhs, R&& rhs) {
        return sad(zero, dx::to_const_mask<S>(mask), __DPL forward<S>(src),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
};

template <typename From, typename To>
concept widening_sad_to = ((signed_integral<From> && signed_integral<To>) ||
                              (unsigned_integral<From> &&
                                  unsigned_integral<To>)) &&
    sizeof(To) > sizeof(From);

template <>
struct fallback_impl<sad_t> {
    // Same element
    template <canonical_vector L, common_vector_with<L> R>
    requires canonical_vector<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr common_canonical_simd_t<L, R> operator()(
        common_canonical_simd_t<L, R> src, L lhs, R rhs) noexcept {
        using sint_t = make_signed_t<simd_element_type_t<L>>;

        constexpr auto abs_diff = [](L lhs, R rhs) noexcept {
            using sint_t = make_signed_t<simd_element_type_t<L>>;
            using E = simd_element_type_t<L>;
            return dx::reinterpret<E>(
                dx::abs(dx::reinterpret<sint_t>(dx::subtract(lhs, rhs))));
        };

        if constexpr (dx::simd_canonical_invocable<hsum_t,
                          common_canonical_simd_t<L, R>>) {
            return dx::add(src, dx::hsum(abs_diff(lhs, rhs)));
        } else {
            // one less broadcast
            return dx::add(
                src, internal::reduction(abs_diff(lhs, rhs), dx::add));
        }
    }

    template <canonical_vector S, canonical_vector T>
    requires same_abi_as<simd_abi_type_t<S>, simd_abi_type_t<T>> &&
        widening_sad_to<simd_element_type_t<T>, simd_element_type_t<S>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr S DPL_VECTORCALL operator()(S src, T lhs, T rhs) noexcept {
        using vidx_t = signed_canonical_vector_t<T>;

        using To = simd_element_type_t<S>;
        using From = simd_element_type_t<T>;
        constexpr auto shift =
            __DPL bit_width(sizeof(To)) - __DPL bit_width(sizeof(From));
        constexpr auto ratio = sizeof(To) / sizeof(From);
        static_assert(__DPL has_single_bit(ratio));

        auto const idx = dx::lane_index<vidx_t>();
        auto const idx_mask = dx::cmplt(idx, simd_abi_traits<S>::size());
        auto const base = dx::bwshift_left(idx_mask, idx, imm<shift>);
        constexpr auto abs_diff = []<typename U>(U lhs, U rhs) noexcept {
            using sint_t = make_signed_t<simd_element_type_t<U>>;
            using E = simd_element_type_t<U>;
            return dx::reinterpret<E>(
                dx::abs(dx::reinterpret<sint_t>(dx::subtract(lhs, rhs))));
        };

        auto addend = abs_diff( //
            dx::element_cast<To>(dx::permute(lhs, base)),
            dx::element_cast<To>(dx::permute(rhs, base)));

        return [&]<size_t I = 1>(this auto self, immediate<I> = imm<1zu>) {
            if constexpr (I == ratio) {
                return dx::add(src, addend);
            } else {
                auto const next_idx =
                    dx::add(idx_mask, base, dx::broadcast<vidx_t>(I));
                auto const next =
                    abs_diff(dx::element_cast<To>(dx::permute(lhs, next_idx)),
                        dx::element_cast<To>(dx::permute(rhs, next_idx)));
                addend = dx::add(next, addend);
                return self(imm<I + 1>);
            }
        }();
    }

    template <simd_vector L, broadcastable_to<L> R, sad_result<L, L> S>
    requires cpo_invocable<sad_t, S, L, canonical_type_t<L>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) noexcept {

        return sad_t::operator()(__DPL forward<S>(src),
            __DPL forward<L>(lhs), dx::broadcast<L>(__DPL forward<R>(rhs)));
    }

    template <simd_vector R, broadcastable_to<R> L, sad_result<R, R> S>
    requires cpo_invocable<sad_t, S, canonical_type_t<R>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, L&& lhs, R&& rhs) noexcept {
        return sad_t::operator()(__DPL forward<S>(src),
            dx::broadcast<R>(__DPL forward<L>(lhs)), __DPL forward<R>(rhs));
    }
};

} // namespace datapar::internal
namespace datapar {
inline namespace cpo {
inline constexpr internal::sad_t sad{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
