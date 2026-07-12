// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/arithmetic/subtract.h"
#include "dpl/core/operations/bit/vrot_vector_for.h"
#include "dpl/core/operations/bitwise.h"
#include "dpl/core/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/from_bitset.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/to_bitset.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/rotate.h"
#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/std/utility/bitset.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rotl(...) noexcept = delete;

struct rotl_t :
    private bit_manipulation_base<rotl_t>,
    private maskable_transform_base<rotl_t> {
    using operation_base<rotl_t>::operator();
    using maskable_transform_base<rotl_t>::operator();

    template <const_mask_like M>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M) noexcept {
        return cmask_v<__DPL rotl(M::value)>;
    }
};

template <>
struct operation_signature<rotl_t> {
    template <simd_type L, integral_constant_like R>
    requires (!simd_vector<R>)
    static consteval void operator()(L&&, R) noexcept {}
    template <simd_type L>
    static consteval void operator()(L&&, size_t) noexcept {}
    template <simd_vector L, simd_vector R>
    static consteval void operator()(L&&, R&&) noexcept {}
};

template <>
struct fallback_impl<rotl_t> {
    template <simd_abi A, simd_element_for<A> E>
    requires integral<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL operator()(
            basic_vector<E, A> val, size_t size) noexcept {
        using ubit = unsigned_representation_t<E>;
        // almost always power of 2, so modulus should optimize to bwand
        constexpr auto digits = sizeof(E) * __DPL char_bit_v;
        size %= digits;
        auto const rsize = digits - size;
        auto const uval = dx::reinterpret<ubit>(val);
        auto const result = dx::bwor(
            dx::bwshift_left(uval, size), dx::bwshift_right(uval, rsize));
        return dx::reinterpret<E>(result);
    }

    template <canonical_mask T>
    requires fixed_width_mask<T> && cpo_invocable<to_bitset_t, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val, size_t size) noexcept {
        using bitset_t = invoke_result_t<to_bitset_t, T>;
        return dx::from_bitset<T>(__DPL rotl(dx::to_bitset(val), size));
    }

    template <simd_abi A, simd_element_for<A> LE, simd_element_for<A> RE>
    requires integral<LE> && integral<RE> && common_size_with<LE, RE> &&
        (__DPL has_single_bit(sizeof(LE)) && __DPL has_single_bit(sizeof(RE)))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<LE, A>
        DPL_VECTORCALL operator()(
            basic_vector<LE, A> val, basic_vector<RE, A> size) noexcept {
        using ubit = unsigned_representation_t<LE>;
        constexpr auto digits = sizeof(LE) * __DPL char_bit_v;
        auto const dig = dx::broadcast<RE, A>(static_cast<RE>(digits));
        size = dx::bwand(size, dx::subtract(dig, 1));
        auto const rsize = dx::subtract(digits, size);
        auto const uval = dx::reinterpret<ubit>(val);
        auto const result = dx::bwor(
            dx::bwshift_left(uval, size), dx::bwshift_right(uval, rsize));
        return dx::reinterpret<LE>(result);
    }
};

template <typename T, typename N = size_t, typename A = simd_abi_type_t<T>>
concept unqualified_canonical_rotl = requires {
    {
        rotl(internal::abi<A>, internal::declarg<T>(), internal::declarg<N>())
    } -> same_as<basic_vector<simd_element_type_t<T>, A>>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_canonical_mrotl = cpo_invocable<rotl_t, T, N> &&
    (!simd_type<S> || same_as<S, cpo_result_t<rotl_t, T, N>>) && requires {
        {
            rotl(internal::abi<cpo_result_t<rotl_t, T, N>>,
                internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> same_as<cpo_result_t<rotl_t, T, N>>;
    };

template <>
struct canonical_impl<rotl_t> {
private:
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
    template <canonical_simd_type T>
    requires unqualified_canonical_rotl<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, size_t count) noexcept {
        return rotl(internal::abi<T>, val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrotl<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val, size_t count) noexcept {
        return rotl(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrotl<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, size_t count) noexcept {
        return rotl(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mrotl<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, size_t count) noexcept {
        return rotl(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mrotl<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, size_t count) noexcept {
        return rotl(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }

    ///
    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_rotl<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, N count) noexcept {
        return rotl(internal::abi<T>, val, count);
    }

    template <canonical_mask T, integral_constant_like N>
    requires unqualified_canonical_rotl<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, N count) noexcept {
        return rotl(internal::abi<T>, val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mrotl<type_identity_t<T>, mask_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val, N count) noexcept {
        return rotl(internal::abi<T>, src, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires unqualified_canonical_mrotl<type_identity_t<T>,
        launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, N count) noexcept {
        return rotl(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, count);
    }

    template <canonical_vector T, integral_constant_like N>
    requires unqualified_canonical_mrotl<dx::zero_t, mask_t<T>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, N count) noexcept {
        return rotl(internal::abi<T>, zero, mask, val, count);
    }

    template <canonical_vector T, const_mask_for<T> M, integral_constant_like N>
    requires canonical_vector<T> &&
        unqualified_canonical_mrotl<dx::zero_t, launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, N count) noexcept {
        return rotl(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, count);
    }

    ///
    template <canonical_vector L, canonical_vrot_vector_for<L> R>
    requires unqualified_canonical_rotl<L, R, common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(L lhs, R rhs) noexcept {
        return rotl(internal::abi<common_abi_t<L, R>>, lhs, rhs);
    }

    template <canonical_vector L, canonical_vrot_vector_for<L> R>
    requires unqualified_canonical_mrotl<vresult_t<L, R>, vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return rotl(internal::abi<common_abi_t<L, R>>, src, mask, lhs, rhs);
    }

    template <canonical_vector L, canonical_vrot_vector_for<L> R,
        const_mask_for<vresult_t<L, R>> M>
    requires unqualified_canonical_mrotl<vresult_t<L, R>,
        launder_cmask_t<vresult_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        vresult_t<L, R> src, M cmask, L lhs, R rhs) noexcept {
        return rotl(internal::abi<common_abi_t<L, R>>, src,
            dx::to_const_mask<vresult_t<L, R>>(cmask), lhs, rhs);
    }

    template <canonical_vector L, canonical_vrot_vector_for<L> R>
    requires unqualified_canonical_mrotl<dx::zero_t, vmask_t<L, R>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, vmask_t<L, R> mask, L lhs, R rhs) noexcept {
        return rotl(internal::abi<common_abi_t<L, R>>, zero, mask, lhs, rhs);
    }

    template <canonical_vector L, canonical_vrot_vector_for<L> R,
        const_mask_for<vresult_t<L, R>> M>
    requires unqualified_canonical_mrotl<dx::zero_t,
        launder_cmask_t<vresult_t<L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr vresult_t<L, R> operator()(
        dx::zero_t zero, M cmask, L lhs, R rhs) noexcept {
        return rotl(internal::abi<common_abi_t<L, R>>, zero,
            dx::to_const_mask<vresult_t<L, R>>(cmask), lhs, rhs);
    }
};

template <typename T, typename N = size_t>
concept unqualified_extended_mask_rotl = requires {
    {
        rotl(internal::declarg<T>(), internal::declarg<N>())
    } -> mask_with_common_abi<simd_abi_type_t<T>>;
};

template <typename T, typename N = size_t, typename A = simd_abi_type_t<T>>
concept unqualified_extended_rotl = requires {
    {
        rotl(internal::declarg<T>(), internal::declarg<N>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T, typename N = size_t>
concept unqualified_extended_mrotl = cpo_invocable<rotl_t, T, N> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<rotl_t, T, N>>) &&
    requires {
        {
            rotl(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<N>())
        } -> equivalent_vector_with<cpo_result_t<rotl_t, T, N>>;
    };

template <>
struct extended_impl<rotl_t> {
public:
    template <extended_mask L>
    requires unqualified_extended_mask_rotl<L>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, size_t shift) {
        return rotl(__DPL forward<L>(lhs), shift);
    }

    template <extended_mask L, integral_constant_like R>
    requires unqualified_extended_mask_rotl<L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R shift) {
        return rotl(__DPL forward<L>(lhs), shift);
    }
    ///

    template <simd_vector L, vrot_vector_for<L> R>
    requires (extended_simd_type<L> || extended_simd_type<R>) &&
        unqualified_extended_rotl<L, R, common_abi_t<L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L&& lhs, R&& rhs) {
        return rotl(__DPL forward<L>(lhs), __DPL forward<L>(rhs));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> L,
        vrot_vector_for<L> R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        unqualified_extended_mrotl<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, L&& lhs, R&& rhs) {
        return rotl(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> L,
        vrot_vector_for<L> R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mrotl<S, launder_cmask_t<S, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, L&& lhs, R&& rhs) {
        return rotl(src, dx::to_const_mask<S>(cmask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, vrot_vector_for<L> R,
        result_mask_for<rotl_t, L, R> M>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mrotl<dx::zero_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, L&& lhs, R&& rhs) {
        return rotl(zero, __DPL forward<M>(mask), __DPL forward<L>(lhs),
            __DPL forward<R>(rhs));
    }

    template <simd_vector L, vrot_vector_for<L> R,
        result_cmask_for<rotl_t, L, R> M>
    requires (extended_vector<L> || extended_vector<R>) &&
        unqualified_extended_mrotl<dx::zero_t,
            launder_cmask_t<cpo_result_t<rotl_t, L, R>, M>, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, L&& lhs, R&& rhs) {
        return rotl(zero, dx::to_const_mask<cpo_result_t<rotl_t, L, R>>(cmask),
            __DPL forward<L>(lhs), __DPL forward<R>(rhs));
    }
    ///
    template <extended_vector T>
    requires unqualified_extended_rotl<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, size_t count) {
        return rotl(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotl<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, size_t count) {
        return rotl( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrotl<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, size_t count) {
        return rotl( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector T, result_mask_for<rotl_t, T, size_t> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotl<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, size_t count) {
        return rotl(zero, mask, __DPL forward<T>(val), count);
    }

    template <simd_vector T, result_cmask_for<rotl_t, T, size_t> M>
    requires extended_vector<T> &&
        unqualified_extended_mrotl<dx::zero_t,
            launder_cmask_t<cpo_result_t<rotl_t, T, size_t>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, size_t count) {
        return rotl(zero,
            dx::to_const_mask<cpo_result_t<rotl_t, T, size_t>>(cmask),
            __DPL forward<T>(val), count);
    }

    ///
    template <extended_vector T, integral_constant_like N>
    requires unqualified_extended_rotl<T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, N count) {
        return rotl(__DPL forward<T>(val), count);
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotl<S, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, N count) {
        return rotl( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T,
        integral_constant_like N>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mrotl<S, launder_cmask_t<S, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, N count) {
        return rotl( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val), count);
    }

    template <simd_vector T, integral_constant_like N,
        result_mask_for<rotl_t, T, N> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mrotl<dx::zero_t, M, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, N count) {
        return rotl(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), count);
    }

    template <extended_vector T, integral_constant_like N,
        result_cmask_for<rotl_t, T, N> M>
    requires unqualified_extended_mrotl<dx::zero_t, launder_cmask_t<T, M>, T, N>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, N count) {
        return rotl(zero, dx::to_const_mask<cpo_result_t<rotl_t, T, N>>(cmask),
            __DPL forward<T>(val), count);
    }
};

template <size_t V>
struct rotli_t {
public:
    template <typename... Args>
    requires cpo_invocable<rotl_t, Args..., immediate<V>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(Args&&... args) noexcept(
        (... && (!simd_type<Args> || canonical_simd_type<Args>))) {
        return rotl_t::operator()(__DPL forward<Args>(args)..., imm<V>);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::rotl_t rotl{};
template <size_t V>
inline constexpr internal::rotli_t<V> rotli{};
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
