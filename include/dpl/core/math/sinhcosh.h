// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/round.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/math/details/ldexp.h"
#  include "dpl/core/math/details/pair.h"
#  include "dpl/core/math/details/polynomial.h"
#  include "dpl/core/operations/bitwise.h" // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void sinh(...) noexcept = delete;
void cosh(...) noexcept = delete;
void sinhcosh(...) noexcept = delete;

struct DPL_EMPTY_BASES sinh_t :
    private math_operation_base<sinh_t>,
    private maskable_transform_base<sinh_t> {
    using math_operation_base<sinh_t>::operator();
    using maskable_transform_base<sinh_t>::operator();
};
struct DPL_EMPTY_BASES cosh_t :
    private math_operation_base<cosh_t>,
    private maskable_transform_base<cosh_t> {
    using math_operation_base<cosh_t>::operator();
    using maskable_transform_base<cosh_t>::operator();
};
struct DPL_EMPTY_BASES sinhcosh_t :
    private math_operation_base<sinhcosh_t>,
    private maskable_transform_base<sinhcosh_t> {
    using math_operation_base<sinhcosh_t>::operator();
    using maskable_transform_base<sinhcosh_t>::operator();
};

template <>
struct operation_signature<sinh_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct operation_signature<cosh_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <>
struct operation_signature<sinhcosh_t> {
    template <simd_vector T, typename O>
    requires exact_mask_for<O, T> || const_mask_for<O, T>
    static consteval void operator()(T&&, O&&) noexcept {}
};

template <auto V>
struct sinhcoshi_t {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_const_mask_t<T, V>;

    template <typename T>
    static constexpr mask_type<T> opmask{};

public:
    template <simd_vector T>
    requires requires { typename mask_type<T>; } &&
        cpo_invocable<sinhcosh_t, T, mask_type<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& arg) noexcept(canonical_vector<T>) {
        return sinhcosh_t::operator()(__DPL forward<T>(arg), opmask<T>);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires requires { typename mask_type<T>; } &&
        cpo_invocable<sinhcosh_t, T, S, M, mask_type<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& arg) noexcept(
        canonical_vector<T> && canonical_vector<S> &&
        (!simd_mask<M> || canonical_mask<M>)) {
        return sinhcosh_t::operator()(__DPL forward<T>(src),
            __DPL forward<T>(mask), __DPL forward<T>(arg), opmask<T>);
    }
    template <typename M, simd_vector T>
    requires requires { typename mask_type<T>; } &&
        cpo_invocable<sinhcosh_t, dx::zero_t, T, M, mask_type<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& arg) noexcept(canonical_vector<T> &&
        (!simd_mask<M> || canonical_mask<M>)) {
        return sinhcosh_t::operator()(
            zero, __DPL forward<T>(mask), __DPL forward<T>(arg), opmask<T>);
    }

    template <typename M, simd_vector T>
    requires requires { typename mask_type<T>; } &&
        cpo_invocable<sinhcosh_t, dx::zero_t, T, M, mask_type<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M&& mask, T&& arg) noexcept(
        canonical_vector<T> && (!simd_mask<M> || canonical_mask<M>)) {
        return operator()(
            dx::zero, __DPL forward<T>(mask), __DPL forward<T>(arg));
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_msinh = cpo_invocable<sinh_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<sinh_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            sinh(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<sinh_t, T>>;
    };

template <>
struct canonical_impl<sinh_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { sinh(internal::abi<A>, val); }
    {
        return sinh(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msinh<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return sinh(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msinh<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return sinh(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msinh<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return sinh(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msinh<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return sinh(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_sinh = requires {
    { sinh(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_msinh = cpo_invocable<sinh_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<sinh_t, T>>) &&
    requires {
        {
            sinh(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<sinh_t, T>>;
    };

template <>
struct extended_impl<sinh_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_sinh<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return sinh(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msinh<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return sinh( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_msinh<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return sinh( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<sinh_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msinh<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return sinh(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<sinh_t, T> M>
    requires unqualified_extended_msinh<dx::zero_t,
        launder_cmask_t<cpo_result_t<sinh_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return sinh(zero, dx::to_const_mask<cpo_result_t<sinh_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mcosh = cpo_invocable<cosh_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<cosh_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            cosh(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<cosh_t, T>>;
    };

template <>
struct canonical_impl<cosh_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept
    requires requires { cosh(internal::abi<A>, val); }
    {
        return cosh(internal::abi<A>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcosh<type_identity_t<T>, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val) noexcept {
        return cosh(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcosh<type_identity_t<T>,
        launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return cosh(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcosh<dx::zero_t, mask_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val) noexcept {
        return cosh(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcosh<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return cosh(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_cosh = requires {
    { cosh(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mcosh = cpo_invocable<cosh_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<cosh_t, T>>) &&
    requires {
        {
            cosh(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<cosh_t, T>>;
    };

template <>
struct extended_impl<cosh_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_cosh<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return cosh(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcosh<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return cosh( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, common_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mcosh<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return cosh( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<cosh_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mcosh<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return cosh(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<cosh_t, T> M>
    requires unqualified_extended_mcosh<dx::zero_t,
        launder_cmask_t<cpo_result_t<cosh_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return cosh(zero, dx::to_const_mask<cpo_result_t<cosh_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <typename T, typename O>
concept unqualified_canonical_sinhcosh = requires(T val, O opmask) {
    { sinhcosh(internal::abi<T>, val, opmask) } -> same_as<T>;
};

template <typename S, typename M, typename T, typename O>
concept unqualified_canonical_msinhcosh = cpo_invocable<sinhcosh_t, T, O> &&
    (!simd_type<S> || same_as<S, cpo_result_t<sinhcosh_t, T, O>>) &&
    requires(S src, M mask, T val, O opmask) {
        {
            sinhcosh(internal::abi<T>, src, mask, val, opmask)
        } -> same_as<cpo_result_t<sinhcosh_t, T, O>>;
    };

template <>
struct canonical_impl<sinhcosh_t> {
private:
    template <typename T>
    using mask_t DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;

public:
    template <fixed_width_vector T, const_mask_for<T> O>
    requires unqualified_canonical_sinhcosh<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(internal::abi<T>, val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> O>
    requires unqualified_canonical_msinhcosh<type_identity_t<T>, mask_t<T>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val, O ops) noexcept {

        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(internal::abi<T>, src, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M, const_mask_for<T> O>
    requires unqualified_canonical_msinhcosh<type_identity_t<T>,
        launder_cmask_t<T, M>, T, launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> O>
    requires unqualified_canonical_msinhcosh<dx::zero_t, mask_t<T>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(internal::abi<T>, zero, mask, val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> M, const_mask_for<T> O>
    requires canonical_vector<T> &&
        unqualified_canonical_msinhcosh<dx::zero_t, launder_cmask_t<T, M>, T,
            launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, opmask);
    }

    ///
    template <fixed_width_vector T, exact_mask_for<T> O>
    requires unqualified_canonical_sinhcosh<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O opmask) noexcept {
        return sinhcosh(internal::abi<T>, val, opmask);
    }

    template <canonical_vector T, exact_mask_for<T> O>
    requires canonical_mask<O> &&
        unqualified_canonical_msinhcosh<type_identity_t<T>, mask_t<T>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, mask_t<T> mask, T val, O opmask) noexcept {
        return sinhcosh(internal::abi<T>, src, mask, val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> M, exact_mask_for<T> O>
    requires canonical_vector<T> && canonical_mask<O> &&
        unqualified_canonical_msinhcosh<type_identity_t<T>,
            launder_cmask_t<T, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, O opmask) noexcept {
        return sinhcosh(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, opmask);
    }

    template <canonical_vector T, exact_mask_for<T> O>
    requires canonical_mask<O> &&
        unqualified_canonical_msinhcosh<dx::zero_t, mask_t<T>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, mask_t<T> mask, T val, O opmask) noexcept {
        return sinhcosh(internal::abi<T>, zero, mask, val, opmask);
    }

    template <fixed_width_vector T, const_mask_for<T> M, exact_mask_for<T> O>
    requires canonical_vector<T> && canonical_mask<O> &&
        unqualified_canonical_msinhcosh<dx::zero_t, launder_cmask_t<T, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, O opmask) noexcept {
        return sinhcosh(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, opmask);
    }
};

template <typename T, typename O, typename A = simd_abi_type_t<T>>
concept unqualified_extended_sinhcosh = requires {
    {
        sinhcosh(internal::declarg<T>(), internal::declarg<O>())
    } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T, typename O>
concept unqualified_extended_msinhcosh = cpo_invocable<sinhcosh_t, T, O> &&
    (!simd_type<S> ||
        equivalent_vector_with<S, cpo_result_t<sinhcosh_t, T, O>>) &&
    requires {
        {
            sinhcosh(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>(), internal::declarg<O>())
        } -> equivalent_vector_with<cpo_result_t<sinhcosh_t, T, O>>;
    };

template <>
struct extended_impl<sinhcosh_t> {
public:
    template <simd_vector T, const_mask_for<T> O>
    requires extended_vector<T> && unqualified_extended_sinhcosh<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(__DPL forward<T>(val), opmask);
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T,
        const_mask_for<T> O>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msinhcosh<S, M, T, launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(__DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), opmask);
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        const_mask_for<T> O>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_msinhcosh<S, launder_cmask_t<S, M>, T,
            launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh( __DPL forward<S>(src), dx::to_const_mask<S>(ops),
            __DPL forward<T>(val), opmask);
    }

    template <simd_vector T, const_mask_for<T> O,
        result_mask_for<sinhcosh_t, T, O> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_msinhcosh<dx::zero_t, M, T, launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(
            zero, __DPL forward<M>(mask), __DPL forward<T>(val), opmask);
    }

    template <simd_vector T, const_mask_for<T> O,
        result_cmask_for<sinhcosh_t, T, O> M>
    requires extended_vector<T> &&
        unqualified_extended_msinhcosh<dx::zero_t,
            launder_cmask_t<cpo_result_t<sinhcosh_t, T, O>, M>, T,
            launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val, O ops) {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(zero,
            dx::to_const_mask<cpo_result_t<sinhcosh_t, T, O>>(cmask),
            __DPL forward<T>(val), opmask);
    }

    ///
    template <simd_vector T, exact_mask_for<T> O>
    requires (extended_vector<T> || extended_vector<O>) &&
        unqualified_extended_sinhcosh<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val, O&& opmask) {
        return sinhcosh(__DPL forward<T>(val), __DPL forward<O>(opmask));
    }

    template <simd_vector S, exact_mask_for<S> M, simd_vector T,
        exact_mask_for<T> O>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T> ||
                 extended_mask<O>) &&
        unqualified_extended_msinhcosh<S, M, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val, O&& opmask) {
        return sinhcosh( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val), __DPL forward<O>(opmask));
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        exact_mask_for<T> O>
    requires (extended_vector<S> || extended_vector<T> || extended_mask<O>) &&
        unqualified_extended_msinhcosh<S, launder_cmask_t<S, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val, O&& opmask) {
        return sinhcosh( __DPL forward<S>(src),
            dx::to_const_mask<launder_cmask_t<S, M>>(cmask),
            __DPL forward<T>(val), __DPL forward<O>(opmask));
    }

    template <simd_vector T, exact_mask_for<T> O,
        result_mask_for<sinhcosh_t, T, O> M>
    requires (extended_mask<M> || extended_vector<T> || extended_mask<O>) &&
        unqualified_extended_msinhcosh<dx::zero_t, M, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M&& mask, T&& val, O&& opmask) {
        return sinhcosh(zero, __DPL forward<M>(mask), __DPL forward<T>(val),
            __DPL forward<O>(opmask));
    }

    template <fixed_width_vector T, exact_mask_for<T> O,
        result_cmask_for<sinhcosh_t, T, O> M>
    requires (extended_vector<T> || extended_mask<O>) &&
        unqualified_canonical_msinhcosh<dx::zero_t,
            launder_cmask_t<cpo_result_t<sinhcosh_t, T, O>, M>, T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t zero, M cmask, T&& val, O&& opmask) {
        return sinhcosh(zero,
            dx::to_const_mask<cpo_result_t<sinhcosh_t, T, O>>(cmask),
            __DPL forward<T>(val), __DPL forward<O>(opmask));
    }
};

template <>
struct fallback_impl<sinhcosh_t> {
private:
    template <floating_point E>
    static constexpr auto polynomial = []() {
        if constexpr (same_as<E, float>) {
            return fmath::polynomial<0.4166637361e-1f, //
                0.8333456703e-2f,                      //
                0.1394256484e-2f,                      //
                0.1980960224e-3f                       //
                >{};
        } else {
            static_assert(same_as<E, double>);
            return fmath::polynomial<0.4166666666666669905e-1,
                0.8333333333333347095e-2, 0.1388888888886763255e-2,
                0.1984126984148071858e-3, 0.2480158735605815065e-4,
                0.2755731892386044373e-5, 0.2755724800902135303e-6,
                0.2505230023782644465e-7, 0.2092255183563157007e-8,
                0.1602472219709932072e-9>{};
        }
    }();

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto is_exp_underflow(basic_vector<E, A> val) noexcept {
        if constexpr (same_as<E, float>) {
            return val < -103.97208f;
        } else {
            static_assert(same_as<E, double>);
            return val < -745.133;
        }
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair<E, A>
        DPL_VECTORCALL exp(basic_vector<E, A> arg) noexcept {
        using simdf = basic_vector<E, A>;
        auto const u = arg * fmath::inv_ln2;
        auto const qf =
            dx::round(u, rounding::to_nearest_int | rounding::no_exc);
        auto const nln2 = []() {
            if constexpr (same_as<float, E>) {
                return fmath::pair<float, A>{
                    .upper = dx::broadcast<A>(-0.693145751953125f),
                    .lower = dx::broadcast<A>(-1.428606765330187045e-06f),
                };
            } else {
                static_assert(same_as<double, E>);
                return fmath::pair<double, A>{
                    .upper = dx::broadcast<A>(
                        -0.69314718055966295651160180568695068359375),
                    .lower = dx::broadcast<A>(
                        -0.28235290563031577122588448175013436025525412068e-12),
                };
            }
        }();

        static_cast<fmath::pair<E, A>>(fmath::ln2);
        auto s = fmath::single(arg) + qf * nln2.upper;
        s = s + qf * nln2.lower;

        auto const poly = polynomial<E>(s.upper);
        auto t = [&]() {
            constexpr E onesixth = static_cast<E>(1.0 / 6.0);
            if constexpr (same_as<E, float>) {
                auto t = s * poly + dx::broadcast<E, A>(onesixth);
                t = s * t + fmath::half;
                t = s + fmath::square(s) * t;
                return fmath::fast(dx::one_v<simdf>) + t;
            } else {
                static_assert(same_as<E, double>);
                auto t = fmath::single(dx::broadcast<simdf>(fmath::half)) +
                    s * dx::broadcast<E, A>(onesixth);
                t = fmath::single(dx::one_v<simdf>) + t * s;
                t = fmath::single(dx::one_v<simdf>) + t * s;
                auto const s4 = [](auto s2) { return s2 * s2; }(
                                    s.upper * s.upper);
                return t + s4;
            }
        }();

        auto const q = dx::element_cast<signed_representation_t<E>>(qf);
        t.upper = fmath::ldexp(fmath::compliance::speed, t.upper, q);
        t.lower = fmath::ldexp(fmath::compliance::speed, t.lower, q);
        auto const underflow = is_exp_underflow(arg);
        t.upper = dx::select(underflow, dx::zero, t.upper);
        t.lower = dx::select(underflow, dx::zero, t.lower);
        return t;
    }

    template <floating_point E>
    static constexpr auto infinity_threshold = []() {
        if constexpr (same_as<E, float>) {
            return 89.0f;
        } else {
            static_assert(same_as<E, double>);
            return 710.0;
        }
    }();

    template <floating_point E, simd_abi A, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A>
        DPL_VECTORCALL fallback(
            basic_vector<E, A> const arg, OpMask opmask) noexcept {
        using simdf = basic_vector<E, A>;

        auto const absarg = dx::abs(arg);
        auto const pair = [&](fmath::pair<E, A> p) {
            auto const inv_p = fmath::rcp(p);
            if constexpr (simd_mask<OpMask>) {
                return p -
                    fmath::make_pair(
                        dx::negate(inv_p.upper, opmask, inv_p.upper),
                        dx::negate(inv_p.lower, opmask, inv_p.lower));
            } else if constexpr (dx::all_of(opmask)) {
                return p + inv_p;
            } else if constexpr (dx::none_of(opmask)) {
                return p - inv_p;
            } else {
                return p -
                    fmath::make_pair(
                        dx::negate(inv_p.upper, opmask, inv_p.upper),
                        dx::negate(inv_p.lower, opmask, inv_p.lower));
            }
        }(exp(absarg));

        auto result = (pair.upper + pair.lower) * fmath::half;
        result = dx::select(
            (absarg <= infinity_threshold<E>) && dx::isfinite(result), result,
            dx::infinity);

        if constexpr (!simd_mask<OpMask>) {
            if constexpr (dx::none_of(opmask)) {
                result = dx::bwxor(result, dx::bwand(arg, dx::msb));
            } else if constexpr (dx::some_of(opmask)) {
                auto const signmask = dx::select(opmask, dx::zero, arg);
                result = dx::bwxor(result, dx::bwand(signmask, dx::msb));
            }
        } else {
            auto const signmask = dx::select(opmask, dx::zero, arg);
            result = dx::bwxor(result, dx::bwand(signmask, dx::msb));
        }

        return dx::select(dx::isnan(arg), dx::all_bits, result);
    }

    template <canonical_vector T, mask_value_t<simd_abi_traits<T>::size> M>
    requires floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto signmask(
        T val, const_mask<simd_abi_traits<T>::size, M> opmask) noexcept {
        if constexpr (dx::none_of(opmask)) {
            return val;
        } else if constexpr (dx::some_of(opmask)) {
            return dx::select(opmask, dx::zero, val);
        } else {
            return dx::all_bits;
        }
    }

public:
    template <canonical_vector T, const_mask_for<T> O>
    requires floating_point<simd_element_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;

        auto const absarg = dx::abs(val);
        auto const pair = [&](fmath::pair<E, A> p) {
            auto const inv_p = fmath::rcp(p);
            if constexpr (dx::all_of(opmask)) {
                return p + inv_p;
            } else if constexpr (dx::none_of(opmask)) {
                return p - inv_p;
            } else {
                return p -
                    fmath::make_pair(
                        dx::negate(inv_p.upper, opmask, inv_p.upper),
                        dx::negate(inv_p.lower, opmask, inv_p.lower));
            }
        }(exp(absarg));

        auto result = (pair.upper + pair.lower) * fmath::half;
        result = dx::select(
            (absarg <= infinity_threshold<E>) && dx::isfinite(result), result,
            dx::infinity);
        auto const smask = signmask(val, opmask);
        if constexpr (simd_type<decltype(smask)>) {
            result = dx::bwxor(result, dx::bwand(smask, dx::msb));
        }

        return dx::select(dx::isnan(val), dx::all_bits, result);
    }

    template <canonical_vector T, exact_mask_for<T> O>
    requires floating_point<simd_element_type_t<T>> && canonical_mask<O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O opmask) noexcept {
        using E = simd_element_type_t<T>;
        using A = simd_abi_type_t<T>;

        auto const absarg = dx::abs(val);
        auto const pair = [&](fmath::pair<E, A> p) {
            auto const inv_p = fmath::rcp(p);
            return p -
                fmath::make_pair(dx::negate(inv_p.upper, opmask, inv_p.upper),
                    dx::negate(inv_p.lower, opmask, inv_p.lower));
        }(exp(absarg));

        auto result = (pair.upper + pair.lower) * fmath::half;
        result = dx::select(
            (absarg <= infinity_threshold<E>) && dx::isfinite(result), result,
            dx::infinity);

        auto const smask = dx::select(opmask, dx::zero, val);
        result = dx::bwxor(result, dx::bwand(smask, dx::msb));
        return dx::select(dx::isnan(val), dx::all_bits, result);
    }
};

template <>
struct fallback_impl<sinh_t> {
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (fixed_width_abi<A>) {
            constexpr auto op = dx::to_const_mask<basic_vector<E, A>>(dx::zero);
            return fallback_impl<sinhcosh_t>::operator()(val, op);
        } else {
            return fallback_impl<sinhcosh_t>::operator()(
                val, dx::broadcast<E, A>(false_type{}));
        }
    }
};

template <>
struct fallback_impl<cosh_t> {
    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (fixed_width_abi<A>) {
            constexpr auto op =
                dx::to_const_mask<basic_vector<E, A>>(dx::all_bits);
            return fallback_impl<sinhcosh_t>::operator()(val, op);
        } else {
            return fallback_impl<sinhcosh_t>::operator()(
                val, dx::broadcast<E, A>(true_type{}));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::sinh_t sinh{};
DPL_EXPORT inline constexpr internal::cosh_t cosh{};
DPL_EXPORT inline constexpr internal::sinhcosh_t sinhcosh{};
DPL_EXPORT template <auto V>
inline constexpr internal::sinhcoshi_t<V> sinhcoshi{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
