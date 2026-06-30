// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/gather.h"
#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/pair.h"
#include "dpl/core/math/internal/polynomial.h"
#include "dpl/core/math/internal/rempi_table.h"
#include "dpl/core/math/round.h"
#include "dpl/core/math/sign.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/inv_pi.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/bitwise.h" // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void sin(...) noexcept = delete;
void cos(...) noexcept = delete;
void sincos(...) noexcept = delete;

struct DPL_EMPTY_BASES sin_t :
    private math_operation_base<sin_t>,
    private maskable_transform_base<sin_t> {
    using math_operation_base<sin_t>::operator();
    using maskable_transform_base<sin_t>::operator();
};
struct DPL_EMPTY_BASES cos_t :
    private math_operation_base<cos_t>,
    private maskable_transform_base<cos_t> {
    using math_operation_base<cos_t>::operator();
    using maskable_transform_base<cos_t>::operator();
};
struct DPL_EMPTY_BASES sincos_t :
    private math_operation_base<sincos_t>,
    private maskable_transform_base<sincos_t> {
    using math_operation_base<sincos_t>::operator();
    using maskable_transform_base<sincos_t>::operator();
};

template <>
struct operation_signature<sin_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct operation_signature<cos_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct operation_signature<sincos_t> {
    template <simd_vector T, typename O>
    requires exact_mask_for<O, T> || const_mask_for<O, T>
    static consteval void operator()(T&&, O&&) noexcept {}
};

template <auto V>
struct sincosi_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_const_mask_t<T, V>;

    template <typename T>
    static constexpr mask_type<T> opmask{};

public:
    template <simd_vector T>
    requires requires { typename mask_type<T>; } &&
        cpo_invocable<sincos_t, T, mask_type<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) noexcept(canonical_vector<T>) {
        return sincos_t::operator()(__DPL forward<T>(arg), opmask<T>);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires requires { typename mask_type<T>; } &&
        cpo_invocable<sincos_t, T, S, M, mask_type<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& arg) noexcept(
        canonical_vector<T> && canonical_vector<S> &&
        (!simd_mask<M> || canonical_mask<M>)) {
        return sincos_t::operator()(__DPL forward<T>(src),
            __DPL forward<T>(mask), __DPL forward<T>(arg), opmask<T>);
    }
    template <typename M, simd_vector T>
    requires requires { typename mask_type<T>; } &&
        cpo_invocable<sincos_t, dx::zero_t, T, M, mask_type<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& arg) noexcept(canonical_vector<T> &&
        (!simd_mask<M> || canonical_mask<M>)) {
        return sincos_t::operator()(
            zero, __DPL forward<T>(mask), __DPL forward<T>(arg), opmask<T>);
    }

    template <typename M, simd_vector T>
    requires requires { typename mask_type<T>; } &&
        cpo_invocable<sincos_t, dx::zero_t, T, M, mask_type<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& arg) noexcept(
        canonical_vector<T> && (!simd_mask<M> || canonical_mask<M>)) {
        return operator()(
            dx::zero, __DPL forward<T>(mask), __DPL forward<T>(arg));
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_msin = cpo_invocable<sin_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<sin_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            sin(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<sin_t, T>>;
    };

template <>
struct canonical_impl<sin_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { sin(internal::abi<A>, val); }
    {
        return sin(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msin<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return sin(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msin<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return sin(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msin<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return sin(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msin<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return sin(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_sin = requires {
    { sin(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_msin = cpo_invocable<sin_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<sin_t, T>>) &&
    requires {
        {
            sin(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<sin_t, T>>;
    };

template <>
struct extended_impl<sin_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_sin<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return sin(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msin<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return sin( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_msin<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return sin( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<sin_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msin<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return sin(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<sin_t, T> M>
    requires unqualified_extended_msin<dx::zero_t,
        launder_cmask_t<cpo_result_t<sin_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return sin(zero, dx::to_const_mask<cpo_result_t<sin_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mcos = cpo_invocable<cos_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<cos_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            cos(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<cos_t, T>>;
    };

template <>
struct canonical_impl<cos_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { cos(internal::abi<A>, val); }
    {
        return cos(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcos<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return cos(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcos<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return cos(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcos<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return cos(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcos<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return cos(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_cos = requires {
    { cos(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mcos = cpo_invocable<cos_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<cos_t, T>>) &&
    requires {
        {
            cos(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<cos_t, T>>;
    };

template <>
struct extended_impl<cos_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_cos<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return cos(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcos<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return cos( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mcos<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return cos( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<cos_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcos<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return cos(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<cos_t, T> M>
    requires unqualified_extended_mcos<dx::zero_t,
        launder_cmask_t<cpo_result_t<cos_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return cos(zero, dx::to_const_mask<cpo_result_t<cos_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <typename T, typename O>
concept unqualified_canonical_sincos = requires(T val, O opmask) {
    { sincos(internal::abi<T>, val, opmask) } -> same_as<T>;
};

template <typename S, typename M, typename T, typename O>
concept unqualified_canonical_msincos = cpo_invocable<sincos_t, T, O> &&
    (!simd_type<S> || same_as<S, cpo_result_t<sincos_t, T, O>>) &&
    requires(S src, M mask, T val, O opmask) {
        {
            sincos(internal::abi<T>, src, mask, val, opmask)
        } -> same_as<cpo_result_t<sincos_t, T, O>>;
    };

template <>
struct canonical_impl<sincos_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <fixed_width_vector T, const_mask_for<T> O>
    requires unqualified_canonical_sincos<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(internal::abi<T>, val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> O>
    requires unqualified_canonical_msincos<type_identity_t<T>, mask_t<T>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val, O ops) noexcept {

        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(internal::abi<T>, src, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M, const_mask_for<T> O>
    requires unqualified_canonical_msincos<type_identity_t<T>,
        launder_cmask_t<T, M>, T, launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> O>
    requires unqualified_canonical_msincos<dx::zero_t, mask_t<T>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(internal::abi<T>, zero, mask, val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> M, const_mask_for<T> O>
    requires canonical_vector<T> &&
        unqualified_canonical_msincos<dx::zero_t, launder_cmask_t<T, M>, T,
            launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, opmask);
    }

    ///
    template <fixed_width_vector T, exact_mask_for<T> O>
    requires unqualified_canonical_sincos<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O opmask) noexcept {
        return sincos(internal::abi<T>, val, opmask);
    }

    template <canonical_vector T, exact_mask_for<T> O>
    requires canonical_mask<O> &&
        unqualified_canonical_msincos<type_identity_t<T>, mask_t<T>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val, O opmask) noexcept {
        return sincos(internal::abi<T>, src, mask, val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> M, exact_mask_for<T> O>
    requires canonical_vector<T> && canonical_mask<O> &&
        unqualified_canonical_msincos<type_identity_t<T>, launder_cmask_t<T, M>,
            T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, O opmask) noexcept {
        return sincos(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, opmask);
    }

    template <canonical_vector T, exact_mask_for<T> O>
    requires canonical_mask<O> &&
        unqualified_canonical_msincos<dx::zero_t, mask_t<T>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, O opmask) noexcept {
        return sincos(internal::abi<T>, zero, mask, val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> M, exact_mask_for<T> O>
    requires canonical_vector<T> && canonical_mask<O> &&
        unqualified_canonical_msincos<dx::zero_t, launder_cmask_t<T, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, O opmask) noexcept {
        return sincos(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, opmask);
    }
};

template <typename T, typename O, typename A = simd_abi_type_t<T>>
concept unqualified_extended_sincos = requires {
    {
        sincos(internal::declarg<T>(), internal::declarg<O>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T, typename O>
concept unqualified_extended_msincos = cpo_invocable<sincos_t, T, O> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<sincos_t, T, O>>) &&
    requires {
        {
            sincos(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<O>())
        } -> equivalent_vector_with<cpo_result_t<sincos_t, T, O>>;
    };

template <>
struct extended_impl<sincos_t> {
public:
    template <simd_vector T, const_mask_for<T> O>
    requires extended_vector<T> && unqualified_extended_sincos<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(__DPL forward<T>(val), opmask);
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T,
        const_mask_for<T> O>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msincos<S, M, T, launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), opmask);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        const_mask_for<T> O>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_msincos<S, launder_cmask_t<S, M>, T,
            launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos( __DPL forward<S>(src), dx::to_const_mask<S>(ops),
            __DPL forward<T>(val), opmask);
    }

    template <simd_vector T, const_mask_for<T> O,
        result_mask_for<sincos_t, T, O> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msincos<dx::zero_t, M, T, launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), opmask);
    }

    template <simd_vector T, const_mask_for<T> O,
        result_cmask_for<sincos_t, T, O> M>
    requires extended_vector<T> &&
        unqualified_extended_msincos<dx::zero_t,
            launder_cmask_t<cpo_result_t<sincos_t, T, O>, M>, T,
            launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(zero,
            dx::to_const_mask<cpo_result_t<sincos_t, T, O>>(cmask),
            __DPL forward<T>(val), opmask);
    }

    ///
    template <simd_vector T, exact_mask_for<T> O>
    requires (extended_vector<T> || extended_vector<O>) &&
        unqualified_extended_sincos<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, O&& opmask) {
        return sincos(__DPL forward<T>(val), __DPL forward<O>(opmask));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T,
        exact_mask_for<T> O>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T> ||
                 extended_mask<O>) &&
        unqualified_extended_msincos<S, M, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, O&& opmask) {
        return sincos( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL forward<O>(opmask));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        exact_mask_for<T> O>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<O>) &&
        unqualified_extended_msincos<S, launder_cmask_t<S, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, O&& opmask) {
        return sincos( __DPL forward<S>(src),
            dx::to_const_mask<launder_cmask_t<S, M>>(cmask),
            __DPL forward<T>(val), __DPL forward<O>(opmask));
    }

    template <simd_vector T, exact_mask_for<T> O,
        result_mask_for<sincos_t, T, O> M>
    requires (extended_mask<M> || extended_vector<T> || extended_mask<O>) &&
        unqualified_extended_msincos<dx::zero_t, M, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, O&& opmask) {
        return sincos(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<O>(opmask));
    }

    template <fixed_width_vector T, exact_mask_for<T> O,
        result_cmask_for<sincos_t, T, O> M>
    requires (extended_vector<T> || extended_mask<O>) &&
        unqualified_canonical_msincos<dx::zero_t,
            launder_cmask_t<cpo_result_t<sincos_t, T, O>, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, O&& opmask) {
        return sincos(zero,
            dx::to_const_mask<cpo_result_t<sincos_t, T, O>>(cmask),
            __DPL forward<T>(val), __DPL forward<O>(opmask));
    }
};

template <>
struct fallback_impl<sincos_t> {
private:
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair<E, A>
        DPL_VECTORCALL pi_pair() noexcept {
        // NOLINTBEGIN
        if constexpr (same_as<float, E>) {
            return fmath::make_pair(                          //
                dx::broadcast<A>(3.1415927410125732422f),     //
                dx::broadcast<A>(-8.7422776573475857731e-08f) //
            );
        } else {
            static_assert(same_as<double, E>);
            return fmath::make_pair(                        //
                dx::broadcast<A>(3.141592653589793116),     //
                dx::broadcast<A>(1.2246467991473532072e-16) //
            );
        }
        // NOLINTEND
    }

    template <floating_point T>
    static constexpr auto polynomial = []() {
        // NOLINTBEGIN
        if constexpr (same_as<float, T>) {
            return fmath::polynomial<-0.166666597127914428710938f, //
                0.00833307858556509017944336f,                     //
                -0.0001981069071916863322258f,                     //
                2.6083159809786593541503e-06f>{};
        } else {
            static_assert(same_as<double, T>);
            return fmath::polynomial<0.00833333333333332974823815, //
                -0.000198412698412696162806809,                    //
                2.75573192239198747630416e-06,                     //
                -2.50521083763502045810755e-08,                    //
                1.60590430605664501629054e-10,                     //
                -7.64712219118158833288484e-13,                    //
                2.81009972710863200091251e-15,                     //
                -7.97255955009037868891952e-18>{};
        }
        // NOLINTEND
    }();

    template <simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<float, A> rempi_low(basic_vector<float, A> qf,
        basic_vector<float, A> arg, OpMask opmask) noexcept {
        constexpr float a = 3.1414794921875f;           // NOLINT
        constexpr float b = 0.00011315941810607910156f; // NOLINT
        constexpr float c = 1.9841872589410058936e-09f; // NOLINT
        using simdf = basic_vector<float, A>;
        auto scale = dx::select(opmask, fmath::half, dx::one_v<simdf>);
        auto sa = a * scale;
        auto sb = b * scale;
        auto sc = c * scale;

        return dx::fnmadd(qf, sc, dx::fnmadd(qf, sb, dx::fnmadd(qf, sa, arg)));
    }

    template <simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<float, A> rempi_mid(basic_vector<float, A> qf,
        basic_vector<float, A> arg, OpMask opmask) noexcept {
        constexpr float a0 = 3.140625f;               // NOLINT
        constexpr float b0 = 0.0009675025939941406f;  // NOLINT
        constexpr float c0 = 1.7881393432617188e-07f; // NOLINT
        constexpr float d0 = -2.781813535079891e-08f; // NOLINT
        constexpr float a1 = 1.5703125f;              // NOLINT
        constexpr float b1 = 0.0004837512969970703f;  // NOLINT
        constexpr float c1 = 5.960464477539063e-08f;  // NOLINT
        constexpr float d1 = 1.5893254712295857e-08f; // NOLINT
        using simdf = basic_vector<float, A>;
        auto sa = dx::select(opmask, dx::broadcast<A>(a1), a0);
        auto sb = dx::select(opmask, dx::broadcast<A>(b1), b0);
        auto sc = dx::select(opmask, dx::broadcast<A>(c1), c0);
        auto sd = dx::select(opmask, dx::broadcast<A>(d1), d0);

        return dx::fnmadd(qf, sd,
            dx::fnmadd(qf, sc, dx::fnmadd(qf, sb, dx::fnmadd(qf, sa, arg))));
    }

    template <simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<double, A> rempi_low(
        basic_vector<double, A> qf, basic_vector<double, A> arg,
        OpMask opmask) noexcept {
        constexpr double a = 3.141592653589793116;      // NOLINT
        constexpr double b = 1.2246467991473532072e-16; // NOLINT
        using simdf = basic_vector<float, A>;

        auto const scale = dx::select(opmask, fmath::half, dx::one_v<simdf>);
        auto const scaled_pi = fmath::scale(pi_pair<double, A>(), scale);

        return dx::fnmadd(
            qf, scaled_pi.lower, dx::fnmadd(qf, scaled_pi.upper, arg));
    }

    template <simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<double, A> rempi_mid(
        basic_vector<double, A> arg, OpMask opmask) noexcept {
        constexpr auto upper_scale = static_cast<double>(1 << 24);
        constexpr auto pi_scale = dx::inv_pi_v<double> / upper_scale;

        fmath::pair<double, A> const dq{
            .upper = dx::trunc(arg * pi_scale, rounding::no_exc) * upper_scale,
            .lower = dx::round(dx::fmsub(arg, dx::inv_pi, dq.upper),
                rounding::to_nearest_int | rounding::no_exc),
        };

        constexpr double a = 3.1415926218032836914;     // NOLINT
        constexpr double b = 3.1786509424591713469e-08; // NOLINT
        constexpr double c = 1.2246467864107188502e-16; // NOLINT
        constexpr double d = 1.2736634327021899816e-24; // NOLINT
        using simdf = basic_vector<float, A>;
        auto scale = dx::select(opmask, fmath::half, dx::one_v<simdf>);
        auto sa = a * scale;
        auto sb = b * scale;
        auto sc = c * scale;
        auto sd = d * scale;

        return dx::fnmadd(dq.lower + dq.upper, d,
            dx::fnmadd(dq.lower, c,
                dx::fnmadd(dq.upper, c,
                    dx::fnmadd(dq.lower, b,
                        dx::fnmadd(dq.upper, b,
                            dx::fnmadd(
                                dq.lower, a, dx::fnmadd(dq.upper, a, arg)))))));
    }

    template <floating_point E>
    static constexpr E threshold_low = []() {
        if constexpr (same_as<float, E>) {
            return 125.0f; // NOLINT
        } else {
            static_assert(same_as<double, E>);
            return 15.0; // NOLINT
        }
    }();

    template <floating_point E>
    static constexpr E threshold_mid = []() {
        if constexpr (same_as<float, E>) {
            return 8.0e6f; // NOLINT
        } else {
            static_assert(same_as<double, E>);
            return 1.0e14; // NOLINT
        }
    }();

    template <floating_point E, simd_abi A>
    struct rempi_single {
        basic_vector<E, A> f;
        basic_vector<signed_representation_t<E>, A> i;
    };
    template <floating_point E, simd_abi A>
    struct rempi_pair {
        fmath::pair<E, A> df;
        basic_vector<signed_representation_t<E>, A> i;
    };

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rempi_single<E, A> quantize_quarters(
        basic_vector<E, A> arg) noexcept {
        // It breaks a value down into its proximity to the nearest quarter
        // (0.25) and identifies which quarter-step it belongs to relative to
        // the nearest whole integer.
        using sint = signed_representation_t<E>;
        constexpr E four = 4.0;
        constexpr E inv_four = 0.25;
        constexpr auto opt = rounding::to_nearest_int | rounding::no_exc;
        auto y = dx::round(arg * four, opt);
        return {
            .f = dx::fnmadd(y, inv_four, arg),
            .i = dx::element_cast<sint>(y - dx::round(arg, opt) * four),
        };
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rempi_pair<E, A> rempi(basic_vector<E, A> arg) noexcept {
        using sint = signed_representation_t<E>;
        using simdi = basic_vector<sint, A>;

        struct expq {
            simdi exp;
            simdi q;
        };

        auto const [exp, q] = [](basic_vector<E, A> arg) {
            auto const n64 = dx::broadcast<sint, A>(-64);
            if constexpr (same_as<E, double>) {
                auto exp = fmath::ilogb(fmath::compliance::unsafe, arg) - 55;
                return expq{
                    .exp = exp,
                    .q = dx::select(exp > (700 - 55), n64, dx::zero),
                };
            } else {
                static_assert(same_as<E, float>);
                auto exp = fmath::ilogb(fmath::compliance::unsafe, arg) - 25;
                return expq{
                    .exp = exp,
                    .q = dx::select(exp > (90 - 25), n64, dx::zero),
                };
            }
        }(arg);

        return [](basic_vector<sint, A> q, basic_vector<E, A> const arg,
                   basic_vector<sint, A> const exp) {
            auto x =
                fmath::single(arg) * mx::gather(fmath::rempi_table<E>, exp);
            auto di = quantize_quarters(x.upper);
            q = di.i;
            x.upper = di.f;
            x = fmath::normalize(x);

            auto y =
                fmath::single(arg) * mx::gather(fmath::rempi_table<E> + 1, exp);
            x = x + y;
            di = quantize_quarters(x.upper);
            q += di.i;
            x.upper = di.f;
            x = fmath::normalize(x);

            using pair = fmath::pair<E, A>;
            y = pair{
                .upper = mx::gather(fmath::rempi_table<E> + 2, exp),
                .lower = mx::gather(fmath::rempi_table<E> + 3, exp),
            };
            y = y * arg;
            x = x + y;
            x = fmath::normalize(x);
            x = x * fmath::scale(pi_pair<E, A>(), 2.0);
            x = fmath::select(dx::abs(arg) < 0.7, fmath::make_pair(arg), x);
            return rempi_pair<E, A>{
                .df = x,
                .i = q,
            };
        }(q, fmath::ldexp(fmath::compliance::unsafe, arg, q),
                   dx::select(exp < dx::zero, dx::zero, exp) << imm<2>);
    }

public:
    template <canonical_vector T, const_mask_for<T> O>
    requires floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O ops) noexcept {
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;
        using sint = signed_representation_t<E>;
        using simdi = basic_vector<sint, A>;
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        auto const qf = [&val, &opmask]() {
            if constexpr (dx::none_of(opmask)) {
                return dx::round(val * dx::inv_pi,
                    rounding::to_nearest_int | rounding::no_exc);
            } else {
                constexpr E half = 0.5;
                constexpr E two = 2.0;
                auto const one = dx::broadcast<T>(dx::one);
                auto const shift =
                    dx::select(opmask, dx::broadcast<T>(half), dx::zero);
                // when evaluating cosine, minus half
                auto const qf = dx::round(dx::fmsub(val, dx::inv_pi, shift),
                    rounding::to_nearest_int | rounding::no_exc);

                auto const a = dx::select(opmask, two, one);
                auto const c = dx::select(opmask, one, dx::zero);
                // when evaluating cos, 2q + 1, when sin, q + 0 (== q)
                // This should be accurate,
                // since the result is only used when
                // magnitude of qf is small < (threshold_mid / pi)
                return dx::fmadd(a, qf, c);
            }
        }();

        auto q = dx::element_cast<sint>(qf);
        auto rem = rempi_low(qf, val, opmask);
        if (auto is_below = dx::abs(val) < threshold_low<E>;
            !dx::all_of(is_below)) {
            rem = dx::select(is_below, rem, rempi_mid(qf, val, opmask));
            if (is_below = dx::abs(val) < threshold_mid<E>;
                !dx::all_of(is_below)) {
                auto dfi = rempi(val);
                auto q2 = dfi.i & 3;
                q2 = q2 + q2 + [&]() {
                    auto const condition = dx::cmple(dfi.df.upper, dx::zero);
                    if constexpr (dx::none_of(opmask)) {
                        auto const base = dx::broadcast<simdi>(2);
                        return dx::subtract(base, condition, base, dx::one);
                    } else if constexpr (dx::all_of(opmask)) {
                        auto const base = dx::broadcast<simdi>(8);
                        return dx::subtract(base, condition, base, dx::one);
                    } else {
                        auto const base = dx::select(opmask,
                            dx::broadcast<simdi>(8), dx::broadcast<simdi>(2));
                        return dx::subtract(base, condition, base, dx::one);
                    }
                }();
                q2 = [&]() {
                    if constexpr (dx::none_of(opmask)) {
                        return q2 >> imm<2>;
                    } else if constexpr (dx::all_of(opmask)) {
                        return q2 >> imm<1>;
                    } else {
                        return dx::select(opmask, q2 >> imm<1>, q2 >> imm<2>);
                    }
                }();

                auto const nhalfpi =
                    fmath::scale(pi_pair<E, A>(), static_cast<E>(-0.5));
                auto const x = dfi.df +
                    fmath::make_pair(                          //
                        dx::sign(nhalfpi.upper, dfi.df.upper), //
                        dx::sign(nhalfpi.lower, dfi.df.upper)  //
                    );
                auto const isoddeven = [&]() {
                    if constexpr (dx::none_of(opmask)) {
                        return (dfi.i & dx::one) == dx::one;
                    } else if constexpr (dx::all_of(opmask)) {
                        return (dfi.i & dx::one) == dx::zero;
                    } else {
                        auto const rhs =
                            dx::select(opmask, dx::zero, dx::one_v<simdi>);
                        return (dfi.i & dx::one) == rhs;
                    }
                }();

                dfi.df = fmath::select(isoddeven, x, dfi.df);
                auto const result = dx::select(
                    dx::isfinite(val), dfi.df.upper + dfi.df.lower, dx::nan);
                q = dx::select(is_below, q, q2);
                rem = dx::select(is_below, rem, result);
            }
        }

        auto const sq_rem = dx::multiply(rem, rem);
        auto const nmask = [&]() {
            if constexpr (dx::none_of(opmask)) {
                return (q & dx::one) == dx::one;
            } else if constexpr (dx::all_of(opmask)) {
                return (q & dx::broadcast<simdi>(2)) == dx::zero;
            } else {
                auto const m =
                    dx::select(opmask, dx::broadcast<simdi>(2), dx::one);
                auto const rhs = dx::select(opmask, dx::zero_v<simdi>, dx::one);
                return (q & m) == rhs;
            }
        }();

        rem = dx::negate(rem, nmask, rem);
        auto const poly = polynomial<E>(sq_rem);
        auto const result = dx::fmadd(sq_rem, poly * rem, rem);
        if constexpr (dx::none_of(opmask)) {
            return dx::select(val == dx::msb, val, result);
        } else if constexpr (dx::all_of(opmask)) {
            return result;
        } else {
            return dx::select(
                val == dx::msb, dx::select(opmask, result, val), result);
        }
    }

    template <canonical_vector T, exact_mask_for<T> O>
    requires floating_point<simd_element_type_t<T>> && canonical_mask<O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O opmask) noexcept {
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;
        using sint = signed_representation_t<E>;
        using simdi = basic_vector<sint, A>;
        auto const qf = [&val, &opmask]() {
            constexpr E half = 0.5;
            constexpr E two = 2.0;
            auto const one = dx::broadcast<T>(dx::one);
            auto const shift =
                dx::select(opmask, dx::broadcast<T>(half), dx::zero);
            // when evaluating cosine, minus half
            auto const qf = dx::round(dx::fmsub(val, dx::inv_pi, shift),
                rounding::to_nearest_int | rounding::no_exc);

            auto const a = dx::select(opmask, two, one);
            auto const c = dx::select(opmask, one, dx::zero);
            // when evaluating cos, 2q + 1, when sin, q + 0 (== q)
            // This should be accurate,
            // since the result is only used when
            // magnitude of qf is small < (threshold_mid / pi)
            return dx::fmadd(a, qf, c);
        }();

        auto const ione = dx::broadcast<simdi>(dx::one);
        auto const itwo = dx::broadcast<simdi>(2);
        auto q = dx::element_cast<sint>(qf);
        auto rem = rempi_low(qf, val, opmask);
        if (auto is_below = dx::abs(val) < threshold_low<E>;
            !dx::all_of(is_below)) {
            rem = dx::select(is_below, rem, rempi_mid(qf, val, opmask));
            if (is_below = dx::abs(val) < threshold_mid<E>;
                !dx::all_of(is_below)) {
                auto dfi = rempi(val);
                auto q2 = dfi.i & 3;
                q2 = q2 + q2 + [&]() {
                    auto const condition = dx::cmple(dfi.df.upper, dx::zero);
                    auto const base =
                        dx::select(opmask, dx::broadcast<simdi>(8), itwo);
                    return dx::subtract(base, condition, base, ione);
                }();
                q2 = dx::select(opmask, q2 >> imm<1>, q2 >> imm<2>);

                auto const nhalfpi =
                    fmath::scale(pi_pair<E, A>(), static_cast<E>(-0.5));
                auto const x = dfi.df +
                    fmath::make_pair(                          //
                        dx::sign(nhalfpi.upper, dfi.df.upper), //
                        dx::sign(nhalfpi.lower, dfi.df.upper)  //
                    );
                auto const isoddeven =
                    (dfi.i & ione) == dx::select(opmask, dx::zero, ione);

                dfi.df = fmath::select(isoddeven, x, dfi.df);
                auto const result = dx::select(dx::isfinite(val),
                    dfi.df.upper + dfi.df.lower, dx::all_bits);
                q = dx::select(is_below, q, q2);
                rem = dx::select(is_below, rem, result);
            }
        }

        auto const sq_rem = dx::multiply(rem, rem);
        auto const nmask = [&]() {
            auto const m = dx::select(opmask, itwo, ione);
            auto const rhs = dx::select(opmask, dx::zero, dx::one_v<simdi>);
            return (q & m) == rhs;
        }();

        rem = dx::negate(rem, nmask, rem);
        auto const result = dx::fmadd(sq_rem, polynomial<E>(sq_rem) * rem, rem);
        return dx::select(
            val == dx::msb, dx::select(opmask, result, val), result);
    }
};

template <>
struct fallback_impl<sin_t> {
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (fixed_width_abi<A>) {
            return fallback_impl<sincos_t>::operator()(val, imm<0b0>);
        } else {
            return fallback_impl<sincos_t>::operator()(
                val, dx::broadcast<E, A>(false_type{}));
        }
    }
};

template <>
struct fallback_impl<cos_t> {
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (fixed_width_abi<A>) {
            constexpr auto op =
                dx::to_const_mask<basic_vector<E, A>>(dx::all_bits);
            return fallback_impl<sincos_t>::operator()(val, op);
        } else {
            return fallback_impl<sincos_t>::operator()(
                val, dx::broadcast<E, A>(true_type{}));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sin_t sin{};
DPL_EXPORT inline constexpr internal::cos_t cos{};
DPL_EXPORT inline constexpr internal::sincos_t sincos{};
DPL_EXPORT template <auto V>
inline constexpr internal::sincosi_t<V> sincosi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
