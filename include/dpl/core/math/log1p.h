// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"
#include "dpl/core/math/log.h"
#include "dpl/core/math/mulx.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/maskable/transform.h"
#  include "dpl/core/dispatch/operation/math.h"
#  include "dpl/core/immediate/constants/infinity.h"
#  include "dpl/core/immediate/constants/nan.h"
#  include "dpl/core/numbers/ext.h"           // IWYU pragma: keep
#  include "dpl/core/operations/arithmetic.h" // IWYU pragma: keep
#  include "dpl/core/operations/bitwise.h"    // IWYU pragma: keep
#  include "dpl/core/operations/compare.h"    // IWYU pragma: keep
#  include "dpl/core/utility/to_tuple_like.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void log1p(...) noexcept = delete;

struct DPL_EMPTY_BASES log1p_t :
    public math_operation_base<log1p_t>,
    public maskable_transform_base<log1p_t> {
    using math_operation_base<log1p_t>::operator();
    using maskable_transform_base<log1p_t>::operator();
};

template <>
struct operation_signature<log1p_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mlog1p = cpo_invocable<log1p_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<log1p_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            log1p(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<log1p_t, T>>;
    };

template <>
struct canonical_impl<log1p_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { log1p(internal::abi<T>, val); }
    {
        return log1p(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mlog1p<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return log1p(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mlog1p<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return log1p(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mlog1p<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return log1p(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mlog1p<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return log1p(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_log1p = requires {
    { log1p(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mlog1p = cpo_invocable<log1p_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<log1p_t, T>>) &&
    requires {
        {
            log1p(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<log1p_t, T>>;
    };

template <>
struct extended_impl<log1p_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_log1p<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return log1p(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mlog1p<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return log1p( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mlog1p<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return log1p( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<log1p_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mlog1p<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return log1p(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<log1p_t, T> M>
    requires unqualified_extended_mlog1p<dx::zero_t,
        launder_cmask_t<cpo_result_t<log1p_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return log1p(zero, dx::to_const_mask<cpo_result_t<log1p_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<log1p_t> {
private:
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL log1p(T val) noexcept {
        using E = simd_element_type_t<T>;
        constexpr auto impl = [](T val, T valp1) {
            using log_impl = fallback_impl<log_t>;
            using sint_t = signed_representation_t<E>;
            auto const [_, exp] =
                dx::to_tuple_like(dx::frexp(valp1, frexp_options::reduced));
            auto const vone = dx::broadcast<T>(dx::one);
            auto const fr = [&] {
                auto const sc =
                    mx::ldexp(mx::compliance::speed, vone, dx::negate(exp));
                return dx::muladd(val, sc, dx::subtract(sc, vone));
            }();

            auto const x =
                fr / mx::pair_ref(mx::single(dx::broadcast<T>(2)) + fr);
            auto const x2 =
                dx::multiply(dx::get_element<0>(x), dx::get_element<0>(x));
            auto const t = log_impl::solve_poly(x2);

            auto const ln2 = log_impl::template ln2_val<T>();
            auto s = mx::pair_ref{ln2} * exp;
            s = mx::pair_ref{s} + mx::scale(x, 2);
            return mx::pair_ref{s} +
                dx::multiply(dx::multiply(dx::get_element<0>(x), x2), t);
        };

        auto const valp1 = dx::add(val, dx::one);
        auto s = [&] {
            if consteval {
                auto const valid = dx::logical_and(dx::cmpgt(valp1, dx::zero),
                    dx::cmple(valp1, dx::max_value));
                auto const cvalp1 = dx::select(valid, valp1, dx::one);
                auto const cval = dx::select(valid, val, dx::one);
                return impl(cval, cvalp1);
            } else {
                return impl(val, valp1);
            }
        }();

        constexpr auto fixflags = fpfix::condition<fpfix::negative, dx::nan> |
            fpfix::condition<fpfix::zero, -dx::infinity> |
            fpfix::condition<fpfix::infinite, fpfix::copy> |
            fpfix::condition<fpfix::nan, fpfix::copy>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            // hope CSE removes the add
            return dx::fixup(valp1, mx::recombine(s), fixflags);
        } else {
            auto result = mx::recombine(s);
            result = dx::select(dx::isinf(val), dx::infinity, result);
            result = dx::select(
                dx::logical_or(dx::cmplt(valp1, dx::zero), dx::isnan(val)),
                dx::all_bits, result);
            result =
                dx::select(dx::cmpeq(valp1, dx::zero), -dx::infinity, result);
            return result;
        }
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, ext::float16> ||
        same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL log1p(T val) noexcept {
        using log_impl = fallback_impl<log_t>;
        using A = simd_abi_type_t<T>;
        constexpr auto log1pf = [](rebind_simd_t<T, float> val,
                                    rebind_simd_t<T, float> valp1) {
            auto [_, exp] =
                dx::to_tuple_like(dx::frexp(valp1, frexp_options::reduced));
            auto const vone = dx::broadcast<A, float>(dx::one);
            auto const fr = [&] {
                auto sc =
                    mx::ldexp(mx::compliance::speed, vone, dx::negate(exp));
                return dx::muladd(val, sc, dx::subtract(sc, vone));
            }();

            auto const x = dx::divide(fr, dx::add(fr, 2));
            auto const x2 = dx::multiply(x, x);
            auto const t = log_impl::solve_poly(x2);

            auto const ln2 = log_impl::template ln2_val<T>();
            auto s = dx::mulacc(dx::multiply(2, x), ln2, exp);
            return dx::mulacc(s, dx::multiply(x2, x), t);
        };

        auto const valp1 = dx::add(val, dx::one);
        auto result = [&] {
            if consteval {
                auto const valid = dx::logical_and(dx::cmpgt(valp1, dx::zero),
                    dx::cmple(valp1, dx::max_value));
                auto const cvalp1 = dx::select(valid, valp1, dx::one);
                auto const cval = dx::select(valid, val, dx::one);
                return mx::apply_widened<float>(log1pf, cval, cvalp1);
            } else {
                return mx::apply_widened<float>(log1pf, val, valp1);
            }
        }();

        constexpr auto fixflags = fpfix::condition<fpfix::negative, dx::nan> |
            fpfix::condition<fpfix::zero, -dx::infinity> |
            fpfix::condition<fpfix::infinite, fpfix::copy> |
            fpfix::condition<fpfix::nan, fpfix::copy>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            return dx::fixup(valp1, result, fixflags);
        } else {
            result = dx::select(dx::isinf(val), dx::infinity, result);
            result = dx::select(
                dx::logical_or(dx::cmplt(valp1, dx::zero), dx::isnan(val)),
                dx::all_bits, result);
            result =
                dx::select(dx::cmpeq(valp1, dx::zero), -dx::infinity, result);
            return result;
        }
    }

public:
    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double> ||
        same_as<simd_element_type_t<T>, ext::float16> ||
        same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL operator()(T val) noexcept {
#if DPL_HAS_CONSTEXPR_BUILTIN(__builtin_log1p)
        if consteval {
            if constexpr (fixed_width_abi<A>) {
                return internal::transform<T>(
                    [](E val) {
                        if constexpr (is_same_v<E, double>) {
                            return __builtin_log1p(val);
                        } else {
                            return __builtin_log1pf(static_cast<float>(val));
                        }
                    },
                    val);
            }
        }
#else
        return fallback_impl::log1p(val);
#endif
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::log1p_t log1p{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
