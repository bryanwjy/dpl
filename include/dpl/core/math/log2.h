// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"

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
void log2(...) noexcept = delete;

struct DPL_EMPTY_BASES log2_t :
    public math_operation_base<log2_t>,
    public maskable_transform_base<log2_t> {
    using math_operation_base<log2_t>::operator();
    using maskable_transform_base<log2_t>::operator();
};

template <>
struct operation_signature<log2_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mlog2 = cpo_invocable<log2_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<log2_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            log2(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<log2_t, T>>;
    };

template <>
struct canonical_impl<log2_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { log2(internal::abi<T>, val); }
    {
        return log2(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mlog2<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return log2(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mlog2<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return log2(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mlog2<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return log2(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mlog2<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return log2(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_log2 = requires {
    { log2(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mlog2 = cpo_invocable<log2_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<log2_t, T>>) &&
    requires {
        {
            log2(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<log2_t, T>>;
    };

template <>
struct extended_impl<log2_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_log2<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return log2(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mlog2<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return log2( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mlog2<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return log2( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<log2_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mlog2<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return log2(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<log2_t, T> M>
    requires unqualified_extended_mlog2<dx::zero_t,
        launder_cmask_t<cpo_result_t<log2_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return log2(zero, dx::to_const_mask<cpo_result_t<log2_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<log2_t> {
private:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL solve_poly(T val) noexcept {
        if constexpr (is_same_v<simd_element_type_t<T>, double>) {
            constexpr mx::polynomial<0.96179669392608091449,
                0.5770780162997058982, 0.4121985945485324709,
                0.3205977477944495502, 0.2623708057488514656,
                0.2200768693152277689, 0.2211941750456081490>
                poly;
            return poly(val);
        } else {
            constexpr mx::polynomial<0.9618012905120f, //
                0.5764790177f,                         //
                0.4374550283f>
                poly;
            return poly(val);
        }
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL log2(T val) noexcept {
        constexpr auto impl = [](T val) {
            auto const [fr, exp] =
                dx::to_tuple_like(dx::frexp(val, frexp_options::reduced));

            auto const one = dx::broadcast<T>(dx::one);
            auto const x =
                mx::pair_ref(fr - mx::single(one)) / (mx::single(one) + fr);
            auto const x2 =
                dx::multiply(dx::get_element<0>(x), dx::get_element<0>(x));
            auto const t = solve_poly(x2);

            auto const inv_halfln2 = []() {
                if constexpr (is_same_v<simd_element_type_t<T>, double>) {
                    return mx::make_pair(dx::broadcast<T>(0x1.71547652b82fep+1),
                        dx::broadcast<T>(0x1.777d0ffda0d24p-55));
                } else {
                    return mx::make_pair(dx::broadcast<T>(0x1.715476p+1f),
                        dx::broadcast<T>(0x1.4ae0cp-25f));
                }
            }();
            auto const x3 = x2 * mx::pair_ref{x};
            auto s = exp + mx::pair_ref{mx::pair_ref{x} * inv_halfln2};
            return s + mx::pair_ref{mx::pair_ref{x3} * t};
        };

        auto s = [&] {
            if consteval {
                return impl(dx::min(dx::abs(val), dx::max_value));
            } else {
                return impl(val);
            }
        }();

        constexpr auto fixflags = fpfix::condition<fpfix::negative, dx::nan> |
            fpfix::condition<fpfix::zero, -dx::infinity> |
            fpfix::condition<fpfix::infinite, fpfix::copy> |
            fpfix::condition<fpfix::nan, fpfix::copy>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            return dx::fixup(val, mx::recombine(s), fixflags);
        } else {
            auto result = mx::recombine(s);
            result = dx::select(dx::isinf(val), dx::infinity, result);
            result = dx::select(
                dx::logical_or(dx::cmplt(val, dx::zero), dx::isnan(val)),
                dx::all_bits, result);
            result =
                dx::select(dx::cmpeq(val, dx::zero), -dx::infinity, result);
            return result;
        }
    }

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, ext::float16> ||
        same_as<simd_element_type_t<T>, ext::bfloat16>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL log2(T val) noexcept {
        using A = simd_abi_type_t<T>;
        constexpr auto log2f = [](rebind_simd_t<T, float> val) {
            auto const [fr, exp] =
                dx::to_tuple_like(dx::frexp(val, frexp_options::reduced));
            auto const vone = dx::broadcast<A, float>(dx::one);
            auto const x =
                dx::divide(dx::subtract(fr, vone), dx::add(fr, vone));
            auto const x2 = dx::multiply(x, x);
            auto const t = solve_poly(x2);

            auto const inv_halfln2 = dx::broadcast<A, float>(0x1.715476p+1f);
            auto s = dx::fmadd(x, inv_halfln2, exp);
            return dx::fmacc(s, dx::multiply(x2, x), t);
        };

        auto const cval = dx::min(dx::abs(val), dx::max_value);
        auto result = mx::apply_widened<float>(log2f, cval);

        constexpr auto fixflags = fpfix::condition<fpfix::negative, dx::nan> |
            fpfix::condition<fpfix::zero, -dx::infinity> |
            fpfix::condition<fpfix::infinite, fpfix::copy> |
            fpfix::condition<fpfix::nan, fpfix::copy>;
        if constexpr (dx::is_simd_canonical_invocable<T, T, decltype(fixflags)>(
                          dx::fixup)) {
            return dx::fixup(val, result, fixflags);
        } else {
            result = dx::select(dx::isinf(val), dx::infinity, result);
            result = dx::select(
                dx::logical_or(dx::cmplt(val, dx::zero), dx::isnan(val)),
                dx::all_bits, result);
            result =
                dx::select(dx::cmpeq(val, dx::zero), -dx::infinity, result);
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
#if DPL_HAS_CONSTEXPR_BUILTIN(__builtin_log2)
        if consteval {
            if constexpr (fixed_width_abi<A>) {
                return internal::transform<T>(
                    [](E val) {
                        if constexpr (is_same_v<E, double>) {
                            return __builtin_log2(val);
                        } else {
                            return __builtin_log2f(static_cast<float>(val));
                        }
                    },
                    val);
            }
        }
#else
        return fallback_impl::log2(val);
#endif
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::log2_t log2{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
