// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/arithmetic/subtract.h"
#include "dpl/core/operations/bit/common.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/compare/cmpeq.h"
#include "dpl/core/operations/compare/cmpneq.h"
#include "dpl/core/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/from_bitset.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_mask_type.h"
#  include "dpl/std/bit/rotate.h"
#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rotr(...) noexcept = delete;

struct rotr_t :
    public bit_manipulation_base<rotr_t>,
    public maskable_transform_base<rotr_t> {
    using operation_base<rotr_t>::operator();
    using maskable_transform_base<rotr_t>::operator();

    template <size_t W, internal::mask_value_t<W> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto operator()(const_mask<W, V> mask) noexcept {
        return const_mask<W, rotr(mask)>{};
    }
};

template <>
struct operation_signature<rotr_t> {
    template <simd_type L, integral_constant_like R>
    requires (!simd_vector<R>)
    static consteval void operator()(L&&, R) noexcept {}
    template <simd_type L>
    static consteval void operator()(L&&, size_t) noexcept {}
    template <simd_vector L, simd_vector R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<rotr_t> {
    template <canonical_vector T>
    requires integral<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val, size_t count) noexcept {
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;
        using uint_t = make_unsigned_t<E>;
        // almost always power of 2, so modulus- should optimize to bwand
        constexpr auto digits = __DPL type_bit_v<uint_t>;
        count %= digits;
        auto const rcount = digits - count;
        auto const uval = dx::reinterpret<uint_t>(val);
        auto const result = [&]() {
            if consteval {
                if (count == 0) {
                    return val;
                } else {
                    return dx::bwor(dx::bwshift_right(uval, count),
                        dx::bwshift_left(uval, rcount));
                }
            } else {
                return dx::bwor(dx::bwshift_right(uval, count),
                    dx::bwshift_left(
                        dx::broadcast<E, A>(count != 0), uval, rcount));
            }
        }();

        return dx::reinterpret<E>(result);
    }

    template <canonical_mask T>
    requires cpo_invocable<to_bitset_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val, size_t size) noexcept {
        using bitset_t = invoke_result_t<to_bitset_t, T>;
        return dx::from_bitset<T>(__DPL rotr(dx::to_bitset(val), size));
    }

    template <canonical_vector L>
    requires integral<simd_element_type_t<L>> &&
        (__DPL has_single_bit(sizeof(simd_element_type_t<L>)))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr L DPL_VECTORCALL operator()(
        L val, unsigned_canonical_vector_t<L> count) noexcept {
        using A = simd_abi_type_t<L>;
        using LE = simd_element_type_t<L>;
        using R = unsigned_canonical_vector_t<L>;
        using uint_t = make_unsigned_t<LE>;
        constexpr auto digits = __DPL type_bit_v<LE>;
        auto const vdigits = dx::broadcast<R>(digits);
        count = dx::bwand(count, dx::subtract(vdigits, 1));
        auto const rcount = dx::subtract(digits, count);
        auto const uval = dx::reinterpret<uint_t>(val);
        auto const result = [&]() {
            if consteval {
                auto const count_iszero = dx::cmpeq(count, dx::zero);
                return dx::bwor(dx::bwshift_right(uval, count),
                    dx::bwshift_left(dx::select(count_iszero, dx::zero, uval),
                        dx::select(count_iszero, dx::zero, rcount)));
            } else {
                auto const count_isnotzero = dx::cmpneq(count, dx::zero);
                return dx::bwor(dx::bwshift_right(uval, count),
                    dx::bwshift_left(count_isnotzero, uval, rcount));
            }
        }();
        return dx::reinterpret<LE>(result);
    }
};

template <typename T, typename N = size_t>
concept unqualified_canonical_rotr = requires {
    {
        rotr(internal::abi<T>, internal::declarg<T>(), internal::declarg<N>())
    } -> same_as<T>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_canonical_mrotr_base =
    cpo_invocable<rotr_t, T, N> && requires {
        {
            rotr(internal::abi<T>, internal::declarg<S>(),
                internal::declarg<M>(), internal::declarg<T>(),
                internal::declarg<N>())
        } -> same_as<T>;
    };

template <typename M, typename T, typename N = size_t>
concept unqualified_canonical_mrotr =
    unqualified_canonical_mrotr_base<T, M, T, N>;

template <typename M, typename T, typename N = size_t>
concept unqualified_canonical_zmrotr =
    unqualified_canonical_mrotr_base<dx::zero_t, M, T, N>;

template <>
struct canonical_impl<rotr_t> {
private:
    template <typename T>
    using vcount_t DPL_NODEBUG = unsigned_canonical_vector_t<T>;

public:
    template <canonical_vector T>
    requires unqualified_canonical_rotr<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, size_t count) noexcept {
        return rotr(internal::abi<T>, val, count);
    }

    template <canonical_mask T>
    requires unqualified_canonical_rotr<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, size_t count) noexcept {
        return rotr(internal::abi<T>, val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrotr<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, size_t count) noexcept {
        return rotr(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrotr<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, size_t count) noexcept {
        return rotr(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zmrotr<simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, size_t count) noexcept {
        return rotr(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zmrotr<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, size_t count) noexcept {
        return rotr(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }

    ///
    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_rotr<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, N count) noexcept {
        return rotr(internal::abi<T>, val, count);
    }

    template <canonical_mask T, integral_constant_like N>
    requires unqualified_canonical_rotr<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, N count) noexcept {
        return rotr(internal::abi<T>, val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mrotr<simd_mask_type_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, N count) noexcept {
        return rotr(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_mrotr<launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, N count) noexcept {
        return rotr(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_zmrotr<simd_mask_type_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, N count) noexcept {
        return rotr(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_zmrotr<launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, N count) noexcept {
        return rotr(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }

    ///
    template <canonical_vector T>
    requires unqualified_canonical_rotr<T, vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T lhs, vcount_t<T> rhs) noexcept {
        return rotr(internal::abi<T>, lhs, rhs);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrotr<simd_mask_type_t<T>, T, vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T lhs, vcount_t<T> rhs) noexcept {
        return rotr(internal::abi<T>, src, mask, lhs, rhs);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrotr<launder_cmask_t<T, M>, T, vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M mask, T lhs, vcount_t<T> rhs) noexcept {
        return rotr(
            internal::abi<T>, src, dx::to_const_mask<T>(mask), lhs, rhs);
    }

    template <canonical_vector T>
    requires unqualified_canonical_zmrotr<simd_mask_type_t<T>, T, vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T lhs, vcount_t<T> rhs) noexcept {
        return rotr(internal::abi<T>, zero, mask, lhs, rhs);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_zmrotr<launder_cmask_t<T, M>, T, vcount_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M mask, T lhs, vcount_t<T> rhs) noexcept {
        return rotr(
            internal::abi<T>, zero, dx::to_const_mask<T>(mask), lhs, rhs);
    }
};

template <typename T, typename N = size_t>
concept unqualified_extended_rotr = requires {
    {
        rotr(internal::declarg<T>(), internal::declarg<N>())
    } -> equivalent_vector_with<T>;
};

template <typename T, typename N = size_t>
concept unqualified_extended_mask_rotr = requires {
    {
        rotr(internal::declarg<T>(), internal::declarg<N>())
    } -> equivalent_mask_with<T>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mrotr_base =
    cpo_invocable<rotr_t, T, N> && requires {
        {
            rotr(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> equivalent_vector_with<T>;
    };

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mrotr =
    unqualified_extended_mrotr_base<S, M, T, N> && equivalent_vector_with<S, T>;

template <typename M, typename T, typename N = size_t>
concept unqualified_extended_zmrotr =
    unqualified_extended_mrotr_base<dx::zero_t, M, T, N>;

template <>
struct extended_impl<rotr_t> {
public:
    template <extended_mask T>
    requires unqualified_extended_rotr<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& lhs, size_t shift) {
        return rotr(__DPL forward<T>(lhs), shift);
    }

    template <extended_mask T, integral_constant_like N>
    requires unqualified_extended_mask_rotr<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& lhs, N shift) {
        return rotr(__DPL forward<T>(lhs), shift);
    }
    ///

    template <simd_vector T, vrot_vector_for<T> N>
    requires (extended_vector<T> || extended_vector<N>) &&
        unqualified_extended_rotr<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& lhs, N&& rhs) {
        return rotr(__DPL forward<T>(lhs), __DPL forward<N>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T,
        vrot_vector_for<T> N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T> ||
                 extended_vector<N>) &&
        unqualified_extended_mrotr<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& lhs, N&& rhs) {
        return rotr(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(lhs), __DPL forward<N>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        vrot_vector_for<T> N>
    requires (extended_vector<S> || extended_vector<T> || extended_vector<N>) &&
        unqualified_extended_mrotr<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& lhs, N&& rhs) {
        return rotr(src, dx::to_const_mask<S>(mask), __DPL forward<T>(lhs),
            __DPL forward<N>(rhs));
    }

    template <simd_vector T, vrot_vector_for<T> N, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T> || extended_vector<N>) &&
        unqualified_extended_zmrotr<M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& lhs, N&& rhs) {
        return rotr(zero, __DPL forward<M>(mask), __DPL forward<T>(lhs),
            __DPL forward<N>(rhs));
    }

    template <simd_vector T, vrot_vector_for<T> N, const_mask_for<T> M>
    requires (extended_vector<T> || extended_vector<N>) &&
        unqualified_extended_zmrotr<launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M mask, T&& lhs, N&& rhs) {
        return rotr(zero, dx::to_const_mask<T>(mask), __DPL forward<T>(lhs),
            __DPL forward<N>(rhs));
    }
    ///
    template <extended_vector T>
    requires unqualified_extended_rotr<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, size_t count) {
        return rotr(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotr<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, size_t count) {
        return rotr( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrotr<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M mask, T&& val, size_t count) {
        return rotr( __DPL forward<S>(src), dx::to_const_mask<S>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector T, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_zmrotr<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, size_t count) {
        return rotr(zero, mask, __DPL forward<T>(val), count);
    }

    template <extended_vector T, const_mask_for<T> M>
    requires unqualified_extended_zmrotr<launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, size_t count) {
        return rotr(
            zero, dx::to_const_mask<T>(cmask), __DPL forward<T>(val), count);
    }

    ///
    template <extended_vector T, integral_constant_like N>
    requires unqualified_extended_rotr<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, N count) {
        return rotr(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotr<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, N count) {
        return rotr( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrotr<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, N count) {
        return rotr( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector T, integral_constant_like N, exact_mask_for<T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_zmrotr<M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, N count) {
        return rotr(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), count);
    }

    template <extended_vector T, integral_constant_like N, const_mask_for<T> M>
    requires unqualified_extended_zmrotr<launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, N count) {
        return rotr(
            zero, dx::to_const_mask<T>(cmask), __DPL forward<T>(val), count);
    }
};

template <size_t V>
struct rotri_t {
public:
    template <typename... Args>
    requires cpo_invocable<rotr_t, Args..., immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Args&&... args) noexcept(
        (... && (!simd_type<Args> || canonical_simd_type<Args>))) {
        return rotr_t::operator()(__DPL forward<Args>(args)..., imm<V>);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::rotr_t rotr{};
template <size_t V>
inline constexpr internal::rotri_t<V> rotri{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
