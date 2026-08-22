// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/round.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/operations/bitwise.h" // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void sinh(...) noexcept = delete;
void cosh(...) noexcept = delete;
void sinhcosh(...) noexcept = delete;

struct DPL_EMPTY_BASES sinh_t :
    public math_operation_base<sinh_t>,
    public maskable_transform_base<sinh_t> {
    using math_operation_base<sinh_t>::operator();
    using maskable_transform_base<sinh_t>::operator();
};
struct DPL_EMPTY_BASES cosh_t :
    public math_operation_base<cosh_t>,
    public maskable_transform_base<cosh_t> {
    using math_operation_base<cosh_t>::operator();
    using maskable_transform_base<cosh_t>::operator();
};
struct DPL_EMPTY_BASES sinhcosh_t :
    public math_operation_base<sinhcosh_t>,
    public maskable_transform_base<sinhcosh_t> {
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
    template <simd_vector T, exact_mask_for<T> O>
    requires extended_vector<T> || extended_mask<O>
    static consteval void operator()(T&&, O&&) noexcept {}

    template <simd_vector T, const_mask_for<T> O>
    static consteval void operator()(T&&, O) noexcept {}

    template <canonical_vector T>
    static consteval void operator()(T, simd_mask_type_t<T>) noexcept {}
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
concept unqualified_canonical_msinh =
    cpo_invocable<sinh_t, T> && requires(S src, M mask, T val) {
        { sinh(internal::abi<T>, src, mask, val) } -> same_as<T>;
    };

template <>
struct canonical_impl<sinh_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { sinh(internal::abi<T>, val); }
    {
        return sinh(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msinh<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return sinh(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msinh<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return sinh(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msinh<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
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

public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { cosh(internal::abi<T>, val); }
    {
        return cosh(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcosh<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return cosh(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcosh<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return cosh(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcosh<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
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
public:
    template <canonical_vector T, const_mask_for<T> O>
    requires unqualified_canonical_sinhcosh<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(internal::abi<T>, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> O>
    requires unqualified_canonical_msinhcosh<T, simd_mask_type_t<T>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, O ops) noexcept {

        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(internal::abi<T>, src, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M, const_mask_for<T> O>
    requires unqualified_canonical_msinhcosh<T, launder_cmask_t<T, M>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> O>
    requires unqualified_canonical_msinhcosh<dx::zero_t, simd_mask_type_t<T>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(internal::abi<T>, zero, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M, const_mask_for<T> O>
    requires unqualified_canonical_msinhcosh<dx::zero_t, launder_cmask_t<T, M>,
        T, launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sinhcosh(
            internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val, opmask);
    }

    ///
    template <canonical_vector T>
    requires unqualified_canonical_sinhcosh<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, simd_mask_type_t<T> opmask) noexcept {
        return sinhcosh(internal::abi<T>, val, opmask);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msinhcosh<T, simd_mask_type_t<T>, T,
        simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, simd_mask_type_t<T> opmask) noexcept {
        return sinhcosh(internal::abi<T>, src, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msinhcosh<T, launder_cmask_t<T, M>, T,
        simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, M cmask, T val,
        simd_mask_type_t<T> opmask) noexcept {
        return sinhcosh(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, opmask);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msinhcosh<dx::zero_t, simd_mask_type_t<T>, T,
        simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, simd_mask_type_t<T> opmask) noexcept {
        return sinhcosh(internal::abi<T>, zero, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msinhcosh<dx::zero_t, launder_cmask_t<T, M>,
        T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, simd_mask_type_t<T> opmask) noexcept {
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
    static constexpr auto rounding_opt =
        rounding::to_nearest_int | rounding::no_exc;

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL solve_polynomial(T val) noexcept {
        if constexpr (is_same_v<simd_element_type_t<T>, double>) {
            constexpr mx::polynomial<0.4166666666666669905e-1,
                0.8333333333333347095e-2, 0.1388888888886763255e-2,
                0.1984126984148071858e-3, 0.2480158735605815065e-4,
                0.2755731892386044373e-5, 0.2755724800902135303e-6,
                0.2505230023782644465e-7, 0.2092255183563157007e-8,
                0.1602472219709932072e-9>
                poly;
            return poly(val);
        } else {
            constexpr mx::polynomial<0.4166637361e-1f, //
                0.8333456703e-2f,                      //
                0.1394256484e-2f,                      //
                0.1980960224e-3f                       //
                >
                poly;
            return poly(val);
        }
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto is_exp_underflow(T val) noexcept {
        if constexpr (is_same_v<simd_element_type_t<T>, double>) {
            return val < -745.133;
        } else {
            static_assert(is_same_v<simd_element_type_t<T>, float>);
            return val < -103.97208f;
        }
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr mx::pair_type_t<T>
        DPL_VECTORCALL exp(T arg) noexcept {
        auto const u = arg * mx::inv_ln2;
        auto const qf = dx::round(u, rounding_opt);
        auto const nln2 = []() {
            if constexpr (is_same_v<simd_element_type_t<T>, double>) {
                return mx::make_pair(
                    dx::broadcast<T>(
                        -0.69314718055966295651160180568695068359375),
                    dx::broadcast<T>(
                        -0.28235290563031577122588448175013436025525412068e-12));
            } else {
                return mx::make_pair( //
                    dx::broadcast<T>(-0.693145751953125f),
                    dx::broadcast<T>(-1.428606765330187045e-06f));
            }
        }();

        auto s = fmath::single(arg) + qf * dx::get_element<0>(nln2);
        s = s + qf * dx::get_element<1>(nln2);

        auto const poly = solve_polynomial(dx::get_element<0>(s));

        auto t = [&]() {
            constexpr auto onesixth =
                static_cast<simd_element_type_t<T>>(1.0 / 6.0);
            auto const vonesixth = dx::broadcast<T>(onesixth);
            auto const vone = dx::broadcast<T>(dx::one);
            auto const vhalf = dx::broadcast<T>(0.5);
            if constexpr (is_same_v<simd_element_type_t<T>, double>) {
                auto t = mx::single(vhalf) + s * vonesixth;
                t = mx::single(vone) + t * s;
                t = mx::single(vone) + t * s;

                auto s4 = dx::get_element<0>(s);
                s4 = dx::multiply(s4, s4);
                s4 = dx::multiply(s4, s4);
                return mx::pair_ref{t} + dx::multiply(s4, u);
            } else {
                auto t = mx::pair_ref{s} * poly + vonesixth;
                t = mx::pair_ref{s} * t + vhalf;
                t = mx::pair_ref{s} + mx::pair_ref{mx::square(s)} * t;
                return mx::fast(vone) + t;
            }
        }();

        using vexp_t = mx::exponent_vector_t<T>;
        using exp_t = simd_element_type_t<vexp_t>;
        auto const q = dx::element_cast<exp_t>(qf);
        t = mx::make_pair(
            mx::ldexp(mx::compliance::speed, dx::get_element<0>(t), q),
            mx::ldexp(mx::compliance::speed, dx::get_element<1>(t), q));

        auto const underflow = is_exp_underflow(arg);
        return mx::make_pair(
            dx::select(underflow, dx::zero, dx::get_element<0>(t)),
            dx::select(underflow, dx::zero, dx::get_element<1>(t)));
    }

    template <canonical_vector T, typename O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL impl(T const arg, O opmask) noexcept {
        constexpr auto const_opmask =
            requires { requires const_mask_like<O> && !dx::some_of(opmask); };

        auto const absarg = dx::abs(arg);
        auto const pair = [&](mx::pair_type_t<T> p) {
            auto const inv_p = mx::rcp(p);
            if constexpr (const_opmask) {
                if constexpr (dx::all_of(opmask)) {
                    return mx::pair_ref{p} + inv_p;
                } else {
                    static_assert(dx::none_of(opmask));
                    return mx::pair_ref{p} - inv_p;
                }
            } else {
                return mx::pair_ref{p} -
                    fmath::make_pair(dx::negate(dx::get_element<0>(inv_p),
                                         opmask, dx::get_element<0>(inv_p)),
                        dx::negate(dx::get_element<1>(inv_p), opmask,
                            dx::get_element<1>(inv_p)));
            }
        }(exp(absarg));

        constexpr auto upper_bound = []() {
            if constexpr (is_same_v<simd_element_type_t<T>, double>) {
                return 710.0;
            } else {
                return 89.0f;
            }
        };

        auto result = dx::multiply(mx::recombine(pair), 0.5);
        result =
            dx::select(dx::cmpgt(absarg, upper_bound), dx::infinity, result);

        if constexpr (const_opmask) {
            if constexpr (dx::none_of(opmask)) {
                result = dx::bwxor(result, dx::bwand(arg, dx::msb));
            }
        } else {
            auto const signmask = dx::select(opmask, dx::zero, arg);
            result = dx::bwxor(result, dx::bwand(signmask, dx::msb));
        }

        return dx::select(dx::isnan(arg), dx::all_bits, result);
    }

public:
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, simd_mask_type_t<T> ops) noexcept {
        return fallback_impl::impl(val, ops);
    }

    template <canonical_vector T, const_mask_for<T> O>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O ops) noexcept {
        constexpr auto cops = dx::to_const_mask<T>(ops);
        return fallback_impl::impl(val, cops);
    }
}; // namespace datapar::internal

template <>
struct fallback_impl<sinh_t> {
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
            constexpr make_const_mask_t<T, 0> op;
            return fallback_impl<sinhcosh_t>::operator()(val, op);
        } else {
            return fallback_impl<sinhcosh_t>::operator()(
                val, dx::broadcast<T>(false_type{}));
        }
    }
};

template <>
struct fallback_impl<cosh_t> {
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
            constexpr make_const_mask_t<T, -1> op;
            return fallback_impl<sinhcosh_t>::operator()(val, op);
        } else {
            return fallback_impl<sinhcosh_t>::operator()(
                val, dx::broadcast<T>(true_type{}));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::sinh_t sinh{};
inline constexpr internal::cosh_t cosh{};
inline constexpr internal::sinhcosh_t sinhcosh{};
template <auto V>
inline constexpr internal::sinhcoshi_t<V> sinhcoshi{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
