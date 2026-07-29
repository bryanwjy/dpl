// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#include "dpl/core/operations/bitwise/vshift_vector_for.h"
#include "dpl/core/operations/internal/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/from_bitset.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/mask_compatibility.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void bwshift_right(...) noexcept = delete;

struct DPL_EMPTY_BASES bwshift_right_t :
    public bitwise_base<bwshift_right_t>,
    public maskable_transform_base<bwshift_right_t> {
    using bitwise_base<bwshift_right_t>::operator();
    using maskable_transform_base<bwshift_right_t>::operator();
};

template <>
struct operation_signature<bwshift_right_t> {
    template <simd_type L, integral_constant_like R>
    requires (!simd_vector<R>)
    static consteval void operator()(L&&, R) noexcept {}
    template <simd_type L>
    static consteval void operator()(L&&, size_t) noexcept {}
    template <simd_vector L, simd_vector R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<bwshift_right_t> {

    template <typename E>
    using bitset_t DPL_NODEBUG = bitset<__DPL type_bit_v<E>>;

    template <fixed_width_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires integral<RE> && integral_bitset_type<bitset_t<LE>> &&
        (sizeof(bitset_t<LE>) == sizeof(LE))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<LE, A> lhs, basic_vector<RE, A> rhs) noexcept {
        return internal::transform<basic_vector<LE, A>>(
            [](auto lhs, auto rhs) {
                if constexpr (signed_integral<LE>) {
                    return static_cast<LE>(lhs >> rhs);
                } else {
                    return __DPL bit_cast<LE>(
                        __DPL bit_cast<bitset_t<LE>>(lhs) >> rhs);
                }
            },
            lhs, rhs);
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    requires integral_bitset_type<bitset_t<E>> &&
        (sizeof(bitset_t<E>) == sizeof(E))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_vector<E, A> val, size_t shift) noexcept {
        return internal::transform<basic_vector<E, A>>(
            [shift](auto lhs) {
                if constexpr (signed_integral<E>) {
                    return static_cast<E>(lhs >> shift);
                } else {
                    return __DPL bit_cast<E>(
                        __DPL bit_cast<bitset_t<E>>(lhs) >> shift);
                }
            },
            val);
    }

    template <fixed_width_abi A, simd_element_for<A> E>
    requires cpo_invocable<to_bitset_t, basic_mask<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        basic_mask<E, A> val, size_t shift) noexcept {
        return dx::from_bitset<E, A>(dx::to_bitset(val) >> shift);
    }
};

template <typename T, typename N = size_t>
concept unqualified_canonical_mask_bwshift_right = requires {
    {
        bwshift_right(
            internal::abi<T>, internal::declarg<T>(), internal::declarg<N>())
    } -> equivalent_mask_with<T>;
};

template <typename T, typename N = size_t, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_bwshift_right = requires {
    {
        bwshift_right(
            internal::abi<A>, internal::declarg<T>(), internal::declarg<N>())
    } -> vector_with<simd_element_type_t<T>, A>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_canonical_mbwshift_right =
    cpo_invocable<bwshift_right_t, T, N> &&
    (!simd_type<S> || same_as<S, cpo_result_t<bwshift_right_t, T, N>>) &&
    requires {
        {
            bwshift_right(internal::abi<cpo_result_t<bwshift_right_t, T, N>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> same_as<cpo_result_t<bwshift_right_t, T, N>>;
    };

template <>
struct canonical_impl<bwshift_right_t> {
private:
    template <typename T, typename A = simd_abi_type_t<T>>
    using result_t DPL_NODEBUG = basic_vector<simd_element_type_t<T>, A>;

    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

    template <typename L, typename R>
    using vresult_t DPL_NODEBUG =
        basic_vector<simd_element_type_t<L>, common_abi_t<L, R>>;

    template <typename L, typename R>
    using vmask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<L>, common_abi_t<L, R>>;

public:
    template <canonical_mask L>
    requires unqualified_canonical_mask_bwshift_right<L, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, size_t rhs) noexcept {
        return bwshift_right(internal::abi<L>, lhs, rhs);
    }

    template <canonical_mask L, integral_constant_like R>
    requires unqualified_canonical_mask_bwshift_right<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, R rhs) noexcept {
        return bwshift_right(internal::abi<L>, lhs, rhs);
    }
    ///

    template <canonical_vector L, canonical_vshift_vector_for<L> R>
    requires unqualified_canonical_bwshift_right<L, R, common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(L lhs, R rhs) noexcept {
        return bwshift_right(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, canonical_vshift_vector_for<L> R>
    requires unqualified_canonical_mbwshift_right<vresult_t<L, R>,
        vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return bwshift_right(
            internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, canonical_vshift_vector_for<L> R,
        const_mask_for<vresult_t<L, R>> M>
    requires unqualified_canonical_mbwshift_right<vresult_t<L, R>,
        launder_cmask_t<vresult_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, M cmask, L lhs, R rhs) noexcept {
        return bwshift_right(internal::abi<common_abi_t<L, R>>, src,
            dx::to_const_mask<vresult_t<L, R>>(cmask), lhs, rhs);
    }

    template <canonical_vector L, canonical_vshift_vector_for<L> R>
    requires unqualified_canonical_mbwshift_right<dx::zero_t, vmask_t<L, R>, L,
        R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return bwshift_right(
            internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, canonical_vshift_vector_for<L> R,
        const_mask_for<vresult_t<L, R>> M>
    requires unqualified_canonical_mbwshift_right<dx::zero_t,
        launder_cmask_t<vresult_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return bwshift_right(internal::abi<common_abi_t<L, R>>, zero,
            dx::to_const_mask<vresult_t<L, R>>(cmask), lhs, rhs);
    }
    ///

    template <canonical_vector L>
    requires unqualified_canonical_bwshift_right<L, size_t>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr L operator()(L lhs, size_t rhs) noexcept {
        return bwshift_right(internal::abi<L>, lhs, rhs);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbwshift_right<result_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, mask_t<T> mask, T val, size_t count) noexcept {
        return bwshift_right(internal::abi<result_t<T>>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mbwshift_right<result_t<T>, launder_cmask_t<T, M>,
            T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, M cmask, T val, size_t count) noexcept {
        return bwshift_right(internal::abi<result_t<T>>, src,
            dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mbwshift_right<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, mask_t<T> mask, T val, size_t count) noexcept {
        return bwshift_right(
            internal::abi<result_t<T>>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_mbwshift_right<dx::zero_t, launder_cmask_t<T, M>,
            T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, M cmask, T val, size_t count) noexcept {
        return bwshift_right(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }

    ///
    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_bwshift_right<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(T val, N count) noexcept {
        return bwshift_right(internal::abi<T>, val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mbwshift_right<result_t<T>, mask_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, mask_t<T> mask, T val, N count) noexcept {
        return bwshift_right(internal::abi<result_t<T>>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mbwshift_right<result_t<T>, launder_cmask_t<T, M>,
            T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        result_t<T> src, M cmask, T val, N count) noexcept {
        return bwshift_right(internal::abi<result_t<T>>, src,
            dx::to_const_mask<result_t<T>>(cmask), val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mbwshift_right<dx::zero_t, mask_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, mask_t<T> mask, T val, N count) noexcept {
        return bwshift_right(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mbwshift_right<dx::zero_t, launder_cmask_t<T, M>,
            T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr result_t<T> operator()(
        dx::zero_t zero, M cmask, T val, N count) noexcept {
        return bwshift_right(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }
};

template <typename T, typename N = size_t>
concept unqualified_extended_mask_bwshift_right = requires {
    {
        bwshift_right(internal::declarg<T>(), internal::declarg<N>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T, typename N = size_t, typename A = simd_abi_type_t<T>>
concept unqualified_extended_bwshift_right = requires {
    {
        bwshift_right(internal::declarg<T>(), internal::declarg<N>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mbwshift_right =
    cpo_invocable<cpo_result_t<bwshift_right_t, T, N>> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<bwshift_right_t, T, N>>) &&
    requires {
        {
            bwshift_right(internal::abi<cpo_result_t<bwshift_right_t, T, N>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> equivalent_vector_with<cpo_result_t<bwshift_right_t, T, N>>;
    };

template <>
struct extended_impl<bwshift_right_t> {
public:
    template <extended_mask L>
    requires unqualified_extended_mask_bwshift_right<L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, size_t shift) {
        return bwshift_right(__DPL forward<L>(lhs), shift);
    }

    template <extended_mask L, integral_constant_like R>
    requires unqualified_extended_mask_bwshift_right<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R shift) {
        return bwshift_right(__DPL forward<L>(lhs), shift);
    }
    ///

    template <simd_vector L, vshift_vector_for<L> R>
    requires (extended_simd_type<L> || extended_simd_type<R>) &&
        unqualified_extended_bwshift_right<L, R, common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return bwshift_right(__DPL forward<L>(lhs), __DPL forward<L>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> L,
        vshift_vector_for<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mbwshift_right<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return bwshift_right(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> L,
        vshift_vector_for<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mbwshift_right<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, L&& lhs, R&& rhs) {
        return bwshift_right(src, dx::to_const_mask<S>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, vshift_vector_for<L> R,
        result_mask_for<bwshift_right_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mbwshift_right<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return bwshift_right(zero, __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector L, vshift_vector_for<L> R,
        result_cmask_for<bwshift_right_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mbwshift_right<dx::zero_t,
            launder_cmask_t<cpo_result_t<bwshift_right_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs) {
        return bwshift_right(zero,
            dx::to_const_mask<cpo_result_t<bwshift_right_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
    ///
    template <extended_vector T>
    requires unqualified_extended_bwshift_right<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, size_t count) {
        return bwshift_right(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, size_t count) {
        return bwshift_right( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, size_t count) {
        return bwshift_right( __DPL forward<S>(src),
            dx::to_const_mask<S>(cmask), __DPL forward<T>(val), count);
    }

    template <simd_vector T, result_mask_for<bwshift_right_t, T, size_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, size_t count) {
        return bwshift_right(zero, mask, __DPL forward<T>(val), count);
    }

    template <simd_vector T, result_cmask_for<bwshift_right_t, T, size_t> M>
    requires extended_vector<T> &&
        unqualified_extended_mbwshift_right<dx::zero_t,
            launder_cmask_t<cpo_result_t<bwshift_right_t, T, size_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, size_t count) {
        return bwshift_right(zero,
            dx::to_const_mask<cpo_result_t<bwshift_right_t, T, size_t>>(cmask),
            __DPL forward<T>(val), count);
    }

    ///
    template <extended_vector T, integral_constant_like N>
    requires unqualified_extended_bwshift_right<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, N count) {
        return bwshift_right(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, N count) {
        return bwshift_right( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, N count) {
        return bwshift_right( __DPL forward<S>(src),
            dx::to_const_mask<S>(cmask), __DPL forward<T>(val), count);
    }

    template <simd_vector T, integral_constant_like N,
        result_mask_for<bwshift_right_t, T, N> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mbwshift_right<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, N count) {
        return bwshift_right(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), count);
    }

    template <extended_vector T, integral_constant_like N,
        result_cmask_for<bwshift_right_t, T, N> M>
    requires unqualified_extended_mbwshift_right<dx::zero_t,
        launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, N count) {
        return bwshift_right(zero,
            dx::to_const_mask<cpo_result_t<bwshift_right_t, T, N>>(cmask),
            __DPL forward<T>(val), count);
    }
};

template <size_t N>
struct bwshift_righti_t {
    template <typename... Ts>
    requires cpo_invocable<bwshift_right_t, Ts..., size_constant<N>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Ts&&... val) noexcept(
        (... && (!simd_type<Ts> || canonical_simd_type<Ts>))) {
        constexpr size_constant<N> shift;
        return bwshift_right_t::operator()(__DPL forward<Ts>(val)..., shift);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::bwshift_right_t bwshift_right{};
template <size_t N>
inline constexpr internal::bwshift_righti_t<N> bwshift_righti{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
