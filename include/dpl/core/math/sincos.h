// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/gather.h"
#include "dpl/core/math/details/ilogb.h"
#include "dpl/core/math/details/ldexp.h"
#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/details/rempi_table.h"
#include "dpl/core/math/round.h"
#include "dpl/core/math/sign.h"
#include "dpl/core/math/trunc.h"

#if !DPL_MODULES
#  include "dpl/core/basic/undefined.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/inv_pi.h"
#  include "dpl/core/immediate/constants/nan.h"
#  include "dpl/core/immediate/constants/zero.h"
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/select.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void sin(...) noexcept = delete;
void cos(...) noexcept = delete;
void sincos(...) noexcept = delete;

struct DPL_EMPTY_BASES sin_t :
    public math_operation_base<sin_t>,
    public maskable_transform_base<sin_t> {
    using math_operation_base<sin_t>::operator();
    using maskable_transform_base<sin_t>::operator();
};
struct DPL_EMPTY_BASES cos_t :
    public math_operation_base<cos_t>,
    public maskable_transform_base<cos_t> {
    using math_operation_base<cos_t>::operator();
    using maskable_transform_base<cos_t>::operator();
};
struct DPL_EMPTY_BASES sincos_t :
    public math_operation_base<sincos_t>,
    public maskable_transform_base<sincos_t> {
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
    requires convertible_to<O, simd_mask_type_t<T>> || const_mask_for<O, T>
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
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { sin(internal::abi<T>, val); }
    {
        return sin(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msin<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return sin(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_msin<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return sin(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msin<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
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
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { cos(internal::abi<T>, val); }
    {
        return cos(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcos<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return cos(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mcos<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return cos(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mcos<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
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
concept unqualified_canonical_msincos =
    cpo_invocable<sincos_t, T, O> && requires(S src, M mask, T val, O opmask) {
        { sincos(internal::abi<T>, src, mask, val, opmask) } -> same_as<T>;
    };

template <>
struct canonical_impl<sincos_t> {
public:
    template <canonical_vector T, const_mask_for<T> O>
    requires unqualified_canonical_sincos<T, O>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(internal::abi<T>, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> O>
    requires unqualified_canonical_msincos<T, simd_mask_type_t<T>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, O ops) noexcept {

        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(internal::abi<T>, src, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M, const_mask_for<T> O>
    requires unqualified_canonical_msincos<T, launder_cmask_t<T, M>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> O>
    requires unqualified_canonical_msincos<dx::zero_t, simd_mask_type_t<T>, T,
        launder_cmask_t<T, O>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val, O ops) noexcept {
        constexpr auto opmask = dx::to_const_mask<T>(ops);
        return sincos(internal::abi<T>, zero, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M, const_mask_for<T> O>
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
    template <canonical_vector T>
    requires unqualified_canonical_sincos<T, simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, simd_mask_type_t<T> opmask) noexcept {
        return sincos(internal::abi<T>, val, opmask);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msincos<T, simd_mask_type_t<T>, T,
        simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src,
        simd_mask_type_t<T> mask, T val, simd_mask_type_t<T> opmask) noexcept {
        return sincos(internal::abi<T>, src, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_msincos<T, launder_cmask_t<T, M>, T,
            simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(type_identity_t<T> src, M cmask, T val,
        simd_mask_type_t<T> opmask) noexcept {
        return sincos(
            internal::abi<T>, src, dx::to_const_mask<T>(cmask), val, opmask);
    }

    template <canonical_vector T>
    requires unqualified_canonical_msincos<dx::zero_t, simd_mask_type_t<T>, T,
        simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, simd_mask_type_t<T> mask,
        T val, simd_mask_type_t<T> opmask) noexcept {
        return sincos(internal::abi<T>, zero, mask, val, opmask);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires canonical_vector<T> &&
        unqualified_canonical_msincos<dx::zero_t, launder_cmask_t<T, M>, T,
            simd_mask_type_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, M cmask, T val, simd_mask_type_t<T> opmask) noexcept {
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
    static constexpr auto round_opt =
        rounding::to_nearest_int | rounding::no_exc;
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr fmath::pair_type_t<T>
        DPL_VECTORCALL make_pi_pair() noexcept {
        // NOLINTBEGIN
        if constexpr (is_same_v<double, simd_element_type_t<T>>) {
            return fmath::make_pair(                        //
                dx::broadcast<T>(3.141592653589793116),     //
                dx::broadcast<T>(1.2246467991473532072e-16) //
            );
        } else {
            return fmath::make_pair(                          //
                dx::broadcast<T>(3.1415927410125732422f),     //
                dx::broadcast<T>(-8.7422776573475857731e-08f) //
            );
        }
        // NOLINTEND
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL solve_poly(T arg) noexcept {
        // NOLINTBEGIN
        if constexpr (is_same_v<double, simd_element_type_t<T>>) {
            mx::polynomial<0.00833333333333332974823815, //
                -0.000198412698412696162806809,          //
                2.75573192239198747630416e-06,           //
                -2.50521083763502045810755e-08,          //
                1.60590430605664501629054e-10,           //
                -7.64712219118158833288484e-13,          //
                2.81009972710863200091251e-15,           //
                -7.97255955009037868891952e-18>
                poly;
            return poly(arg);
        } else {
            constexpr mx::polynomial<-0.166666597127914428710938f, //
                0.00833307858556509017944336f,                     //
                -0.0001981069071916863322258f,                     //
                2.6083159809786593541503e-06f>
                poly;
            return poly(arg);
        }
        // NOLINTEND
    }

    template <canonical_vector T>
    using vquadrant_t DPL_NODEBUG =
        rebind_simd_t<T, signed_representation_t<simd_element_type_t<T>>>;

    template <canonical_vector T, typename OpMask>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T rempi_low(T qf, T arg, OpMask opmask) noexcept {
        if constexpr (is_same_v<simd_element_type_t<T>, double>) {
            constexpr double a = 3.141592653589793116;      // NOLINT
            constexpr double b = 1.2246467991473532072e-16; // NOLINT

            auto const scale = dx::select(opmask, mx::half, dx::one_v<T>);
            auto const scaled_pi = fmath::scale(make_pi_pair<T>(), scale);

            return dx::nmuladd(
                qf, scaled_pi.lower, dx::nmuladd(qf, scaled_pi.upper, arg));
        } else {
            constexpr float a = 3.1414794921875f;           // NOLINT
            constexpr float b = 0.00011315941810607910156f; // NOLINT
            constexpr float c = 1.9841872589410058936e-09f; // NOLINT
            auto scale = dx::select(opmask, mx::half, dx::one_v<T>);
            auto sa = a * scale;
            auto sb = b * scale;
            auto sc = c * scale;

            return dx::nmuladd(
                qf, sc, dx::nmuladd(qf, sb, dx::nmuladd(qf, sa, arg)));
        }
    }

    template <canonical_vector T, typename OpMask>
    requires same_as<simd_element_type_t<T>, float>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T rempi_mid(T qf, T arg, OpMask opmask) noexcept {
        constexpr float a0 = 3.140625f;               // NOLINT
        constexpr float b0 = 0.0009675025939941406f;  // NOLINT
        constexpr float c0 = 1.7881393432617188e-07f; // NOLINT
        constexpr float d0 = -2.781813535079891e-08f; // NOLINT

        // somehow these constants is more well behaved for cosf
        constexpr float a1 = 1.5703125f;              // NOLINT
        constexpr float b1 = 0.0004837512969970703f;  // NOLINT
        constexpr float c1 = 5.960464477539063e-08f;  // NOLINT
        constexpr float d1 = 1.5893254712295857e-08f; // NOLINT

        auto sa =
            dx::select(opmask, dx::broadcast<T>(a1), dx::broadcast<T>(a0));
        auto sb =
            dx::select(opmask, dx::broadcast<T>(b1), dx::broadcast<T>(b0));
        auto sc =
            dx::select(opmask, dx::broadcast<T>(c1), dx::broadcast<T>(c0));
        auto sd =
            dx::select(opmask, dx::broadcast<T>(d1), dx::broadcast<T>(d0));

        return dx::nmuladd(qf, sd,
            dx::nmuladd(qf, sc, dx::nmuladd(qf, sb, dx::nmuladd(qf, sa, arg))));
    }

    template <canonical_vector T, typename OpMask>
    requires same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T rempi_mid(T arg, OpMask opmask) noexcept {
        constexpr auto upper_scale =
            static_cast<double>(1ll << floating_point_traits<double>::digits);
        constexpr auto pi_scale = dx::inv_pi_v<double> / upper_scale;

        auto const dq = [&]() {
            auto const upper = dx::multiply(
                dx::trunc(dx::multiply(arg, pi_scale), rounding::no_exc),
                upper_scale);
            return mx::make_pair(upper,
                dx::round(dx::mulsub(arg, dx::inv_pi, upper), round_opt));
        }();

        constexpr double a0 = 3.14159262180328369141e+00; // NOLINT
        constexpr double b0 = 3.17865094245917134685e-08; // NOLINT
        constexpr double c0 = 1.22464678641071885020e-16; // NOLINT
        constexpr double d0 = 1.27366343270218998159e-24; // NOLINT
        auto scale = dx::select(opmask, fmath::half, dx::one_v<T>);
        auto sa = a0 * scale;
        auto sb = b0 * scale;
        auto sc = c0 * scale;
        auto sd = d0 * scale;

        auto const [upper, lower] = dq;
        return dx::nmuladd(mx::recombine(dq), sd,
            dx::nmuladd(lower, sc,
                dx::nmuladd(upper, sc,
                    dx::nmuladd(lower, sb,
                        dx::nmuladd(upper, sb,
                            dx::nmuladd(
                                lower, sa, dx::nmuladd(upper, sa, arg)))))));
    }

    template <floating_point_like E>
    static constexpr E threshold_low = []() {
        if constexpr (is_same_v<float, E>) {
            return 125.0f; // NOLINT
        } else {
            static_assert(is_same_v<double, E>);
            return 15.0; // NOLINT
        }
    }();

    template <floating_point_like E>
    static constexpr E threshold_mid = []() {
        if constexpr (is_same_v<float, E>) {
            return 8.0e6f; // NOLINT
        } else {
            static_assert(is_same_v<double, E>);
            return 1.0e14; // NOLINT
        }
    }();

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T quantize_quarters(
        T arg, mx::exponent_vector_t<T>& idx) noexcept {
        // It breaks a value down into its proximity to the nearest quarter
        // (0.25) and identifies which quarter-step it belongs to relative to
        // the nearest whole integer.
        using sint_t = simd_element_type_t<mx::exponent_vector_t<T>>;
        auto const vfour = dx::broadcast<T>(4.0);
        auto const vquart = dx::broadcast<T>(0.25);
        auto const y = dx::round(dx::multiply(arg, vfour), round_opt);
        auto const rarg = dx::round(arg, round_opt);
        idx = dx::element_cast<sint_t>(dx::nmuladd(rarg, vfour, y));
        return dx::nmuladd(y, vquart, arg);
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T rempi(T arg, mx::exponent_vector_t<T>& idx) noexcept {
        using exp_t = simd_element_type_t<mx::exponent_vector_t<T>>;
        using E = simd_element_type_t<T>;
        constexpr exp_t exp_scale = is_same_v<E, double> ? 55 : 25;
        constexpr exp_t q_scale = is_same_v<E, double> ? (700 - 55) : (90 - 25);
        auto const n64 = dx::broadcast<exp_t>(-64);
        auto exp = mx::ilogb(mx::compliance::unsafe, arg) - exp_scale;
        arg = mx::ldexp(
            mx::compliance::unsafe, arg, dx::cmpgt(exp, q_scale), arg, n64);
        exp = dx::bwshift_left(dx::cmpgt(exp, dx::zero), exp, imm<2>);

        return [&idx](T const arg, mx::exponent_vector_t<T> const exp) {
            auto x = mx::single(arg) * mx::gather(mx::rempi_table<E>, exp);
            auto y = mx::single(arg) * mx::gather(mx::rempi_table<E> + 1, exp);

            auto qidx = dx::undefined<mx::exponent_vector_t<T>>();
            auto dif = quantize_quarters(dx::get_element<0>(x), qidx);
            x = mx::normalize(dx::set_element<0>(x, dif));
            x = x + y;

            idx = qidx;
            dif = quantize_quarters(dx::get_element<0>(x), qidx);
            x = mx::normalize(dx::set_element<0>(x, dif));

            idx = dx::add(idx, qidx); // Write to result

            y = mx::make_pair( //
                mx::gather(mx::rempi_table<E> + 2, exp),
                mx::gather(mx::rempi_table<E> + 3, exp));
            y = mx::pair_ref{y} * arg;

            x = mx::normalize(x + y);
            x = x * mx::scale(make_pi_pair<T>(), 2.0);
            x = mx::select(dx::cmplt(dx::abs(arg), 0.7), mx::make_pair(arg), x);
            return x;
        }(arg, exp);
    }

public:
    template <canonical_vector T, typename O>
    requires (canonical_mask<O> && exact_mask_for<O, T> ||
                 const_mask_for<O, T>) &&
        (same_as<simd_element_type_t<T>, float> ||
            same_as<simd_element_type_t<T>, double>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val, O ops) noexcept {
        using E = simd_element_type_t<T>;
        using vexp_t = mx::exponent_vector_t<T>;
        using sint_t = simd_element_type_t<vexp_t>;
        constexpr auto const_opmask = requires {
            requires const_mask_like<O> &&
                !dx::some_of(dx::to_const_mask<T>(ops));
        };

        auto const qf = [&]() {
            if constexpr (const_opmask) {
                constexpr auto opmask = dx::to_const_mask<T>(ops);
                if constexpr (dx::none_of(opmask)) {
                    return dx::round(dx::multiply(val, dx::inv_pi), round_opt);
                } else {
                    auto const vhalf = dx::broadcast<T>(0.5);
                    auto const vtwo = dx::broadcast<T>(2.0);
                    auto const vone = dx::broadcast<T>(dx::one);
                    auto const qf = dx::round(
                        dx::mulsub(val, dx::inv_pi, vhalf), round_opt);
                    return dx::muladd(vtwo, qf, vone);
                }
            } else {
                auto const vhalf = dx::broadcast<T>(0.5);
                auto const vtwo = dx::broadcast<T>(2.0);
                auto const vone = dx::broadcast<T>(dx::one);
                auto const shift = dx::select(ops, vhalf, dx::zero);
                // when evaluating cosine, minus half
                auto const qf =
                    dx::round(dx::mulsub(val, dx::inv_pi, shift), round_opt);
                auto const a = dx::select(ops, vtwo, vone);
                auto const c = dx::select(ops, vone, dx::zero);
                // when evaluating cos, 2q + 1, when sin, q + 0 (== q)
                // This should be accurate,
                // since the result is only used when
                // magnitude of qf is small < (threshold_mid / pi)
                return dx::muladd(a, qf, c);
            }
        }();

        auto q = dx::element_cast<sint_t>(qf);
        auto rem = rempi_low(qf, val, ops);
        if (auto is_below = dx::abs(val) < threshold_low<E>;
            !dx::all_of(is_below)) {
            rem = dx::select(is_below, rem, rempi_mid(qf, val, ops));
            if (is_below = dx::abs(val) < threshold_mid<E>;
                !dx::all_of(is_below)) {

                auto rempi_i = dx::undefined<mx::exponent_vector_t<T>>();
                auto rempi_df = rempi(val, rempi_i);
                auto q2 = dx::bwand(rempi_i, 3);
                q2 = dx::add(q2, q2);

                {
                    auto const condition =
                        dx::cmple(dx::get_element<0>(rempi_df), dx::zero);
                    if constexpr (const_opmask) {
                        constexpr auto opmask = dx::to_const_mask<T>(ops);
                        if constexpr (dx::none_of(opmask)) {
                            auto const base = dx::broadcast<vexp_t>(2);
                            q2 = dx::add(q2,
                                dx::subtract(base, condition, base, dx::one));
                            q2 = dx::bwshift_right(q2, imm<2zu>);
                        } else {
                            static_assert(dx::all_of(opmask));
                            auto const base = dx::broadcast<vexp_t>(8);
                            q2 = dx::add(q2,
                                dx::subtract(base, condition, base, dx::one));
                            q2 = dx::bwshift_right(q2, imm<1zu>);
                        }
                    } else {
                        auto const base = dx::select(ops,
                            dx::broadcast<vexp_t>(8), dx::broadcast<vexp_t>(2));
                        q2 = dx::add(
                            q2, dx::subtract(base, condition, base, dx::one));
                        q2 = dx::select(ops, dx::bwshift_right(q2, imm<1zu>),
                            dx::bwshift_right(q2, imm<2zu>));
                    }
                }

                auto const [nhalfpi_u, nhalfpi_l] = dx::to_tuple_like(
                    mx::scale(make_pi_pair<T>(), static_cast<E>(-0.5)));
                auto const x = mx::pair_ref{rempi_df} +
                    mx::make_pair(                                         //
                        dx::sign(nhalfpi_u, dx::get_element<0>(rempi_df)), //
                        dx::sign(nhalfpi_l, dx::get_element<0>(rempi_df))  //
                    );

                auto const isoddeven = [&]() {
                    if constexpr (const_opmask) {
                        constexpr auto opmask = dx::to_const_mask<T>(ops);
                        if constexpr (dx::none_of(opmask)) {
                            return dx::bwand(rempi_i, dx::one) == dx::one;
                        } else {
                            return dx::bwand(rempi_i, dx::one) == dx::zero;
                        }
                    } else {
                        auto const rhs =
                            dx::select(ops, dx::zero, dx::one_v<vexp_t>);
                        return dx::bwand(rempi_i, dx::one) == rhs;
                    }
                }();

                rempi_df = mx::select(isoddeven, x, rempi_df);
                auto const result = dx::select(
                    dx::isfinite(val), mx::recombine(rempi_df), dx::nan);
                q = dx::select(is_below, q, q2);
                rem = dx::select(is_below, rem, result);
            }
        }

        auto const sq_rem = dx::multiply(rem, rem);
        auto const nmask = [&]() {
            if constexpr (const_opmask) {
                constexpr auto opmask = dx::to_const_mask<T>(ops);
                if constexpr (dx::none_of(opmask)) {
                    return dx::cmpeq(dx::bwand(q, dx::one), dx::one);
                } else {
                    return dx::cmpeq(
                        dx::bwand(q, dx::broadcast<vexp_t>(2)), dx::zero);
                }
            } else {
                auto const m =
                    dx::select(ops, dx::broadcast<vexp_t>(2), dx::one);
                auto const rhs = dx::select(ops, dx::zero_v<vexp_t>, dx::one);
                return dx::cmpeq(dx::bwand(q, m), rhs);
            }
        }();

        rem = dx::negate(rem, nmask, rem);
        auto const poly = solve_poly(sq_rem);
        auto const result = dx::muladd(sq_rem, dx::multiply(poly, rem), rem);
        if constexpr (const_opmask) {
            constexpr auto opmask = dx::to_const_mask<T>(ops);
            if constexpr (dx::none_of(opmask)) {
                return dx::select(dx::cmpeq(val, dx::zero), dx::zero, result);
            } else {
                return result;
            }
        } else {
            return dx::select(
                dx::cmpeq(val, dx::zero), dx::select(ops, result, val), result);
        }
    }
};

template <>
struct fallback_impl<sin_t> {
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
            constexpr make_const_mask_t<T, 0> op;
            return fallback_impl<sincos_t>::operator()(val, op);
        } else {
            return fallback_impl<sincos_t>::operator()(
                val, dx::broadcast<T>(false_type{}));
        }
    }
};

template <>
struct fallback_impl<cos_t> {
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T operator()(T val) noexcept {
        if constexpr (fixed_width_abi<simd_abi_type_t<T>>) {
            constexpr make_const_mask_t<T, -1> op;
            return fallback_impl<sincos_t>::operator()(val, op);
        } else {
            return fallback_impl<sincos_t>::operator()(
                val, dx::broadcast<T>(true_type{}));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::sin_t sin{};
inline constexpr internal::cos_t cos{};
inline constexpr internal::sincos_t sincos{};
template <auto V>
inline constexpr internal::sincosi_t<V> sincosi{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
