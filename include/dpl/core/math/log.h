// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/pair.h"
#include "dpl/core/math/details/polynomial.h"
#include "dpl/core/math/fixup.h"
#include "dpl/core/math/frexp.h"
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
void log(...) noexcept = delete;
struct log1p_t;

struct DPL_EMPTY_BASES log_t :
    public math_operation_base<log_t>,
    public maskable_transform_base<log_t> {
    using math_operation_base<log_t>::operator();
    using maskable_transform_base<log_t>::operator();
};

template <>
struct operation_signature<log_t> {
    static consteval void operator()(simd_vector auto&&) noexcept {}
};

template <typename S, typename M, typename T>
concept unqualified_canonical_mlog = cpo_invocable<log_t, T> &&
    (!simd_type<S> || same_as<S, cpo_result_t<log_t, T>>) &&
    requires(S src, M mask, T val) {
        {
            log(internal::abi<T>, src, mask, val)
        } -> same_as<cpo_result_t<log_t, T>>;
    };

template <>
struct canonical_impl<log_t> {
public:
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(T val) noexcept
    requires requires { log(internal::abi<T>, val); }
    {
        return log(internal::abi<T>, val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mlog<T, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, simd_mask_type_t<T> mask, T val) noexcept {
        return log(internal::abi<T>, src, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mlog<T, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        type_identity_t<T> src, M cmask, T val) noexcept {
        return log(internal::abi<T>, src, dx::to_const_mask<T>(cmask), val);
    }

    template <canonical_vector T>
    requires unqualified_canonical_mlog<dx::zero_t, simd_mask_type_t<T>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(
        dx::zero_t zero, simd_mask_type_t<T> mask, T val) noexcept {
        return log(internal::abi<T>, zero, mask, val);
    }

    template <canonical_vector T, const_mask_for<T> M>
    requires unqualified_canonical_mlog<dx::zero_t, launder_cmask_t<T, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(dx::zero_t zero, M cmask, T val) noexcept {
        return log(internal::abi<T>, zero, dx::to_const_mask<T>(cmask), val);
    }
};

template <typename T, typename A = simd_abi_type_t<T>>
concept unqualified_extended_log = requires {
    { log(internal::declarg<T>()) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T>
concept unqualified_extended_mlog = cpo_invocable<log_t, T> &&
    (!simd_type<S> || equivalent_vector_with<S, cpo_result_t<log_t, T>>) &&
    requires {
        {
            log(internal::declarg<S>(), internal::declarg<M>(),
                internal::declarg<T>())
        } -> equivalent_vector_with<cpo_result_t<log_t, T>>;
    };

template <>
struct extended_impl<log_t> {
public:
    template <extended_vector T>
    requires unqualified_extended_log<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return log(__DPL forward<T>(val));
    }

    template <simd_vector S, exact_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mlog<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M&& mask, T&& val) {
        return log( __DPL forward<S>(src), __DPL forward<M>(mask),
            __DPL forward<T>(val));
    }

    template <simd_vector S, const_mask_for<S> M, equivalent_vector_with<S> T>
    requires (extended_vector<S> || extended_vector<T>) &&
        unqualified_extended_mlog<S, launder_cmask_t<S, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S&& src, M cmask, T&& val) {
        return log( __DPL forward<S>(src), dx::to_const_mask<S>(cmask),
            __DPL forward<T>(val));
    }

    template <simd_vector T, result_mask_for<log_t, T> M>
    requires (extended_mask<M> || extended_vector<T>) &&
        unqualified_extended_mlog<dx::zero_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M&& mask, T&& val) {
        return log(zero, __DPL forward<M>(mask), __DPL forward<T>(val));
    }

    template <extended_vector T, result_cmask_for<log_t, T> M>
    requires unqualified_extended_mlog<dx::zero_t,
        launder_cmask_t<cpo_result_t<log_t, T>, M>, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t zero, M cmask, T&& val) {
        return log(zero, dx::to_const_mask<cpo_result_t<log_t, T>>(cmask),
            __DPL forward<T>(val));
    }
};

template <>
struct fallback_impl<log_t> {
private:
    friend fallback_impl<log1p_t>;
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL solve_poly(T val) noexcept {
        if constexpr (is_same_v<simd_element_type_t<T>, double>) {
            constexpr mx::polynomial< //
                0x1.555555555556ep-1, // 0.6666666666666694
                0x1.999999997c6ddp-2, // 0.3999999999933673
                0x1.24924951e6b74p-2, // 0.28571428835282764
                0x1.c71c3d0112d0cp-3, // 0.22222182902837806
                0x1.746bafa8c6bap-3,  // 0.1818460200673515
                0x1.3903d5e459399p-3, // 0.15283934691538262
                0x1.35943ced4a8e4p-3  // 0.15116164778857588
                >
                poly;
            return poly(val);
        } else {
            constexpr mx::polynomial<0x1.555566p-1f, // 0.66666716
                0x1.99622ep-2f,                      // 0.3997886
                0x1.32c936p-2f                       // 0.29959568
                >
                poly;
            return poly(val);
        }
    }

    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto ln2_val() noexcept {
        if constexpr (is_same_v<simd_element_type_t<T>, double>) {
            return mx::make_pair(dx::broadcast<T>(0x1.62e42fefa39efp-1),
                dx::broadcast<T>(0x1.abc9e3b39803fp-56));
        } else if constexpr (is_same_v<simd_element_type_t<T>, float>) {
            return mx::make_pair(dx::broadcast<T>(0x1.62e43p-1f),
                dx::broadcast<T>(-0x1.05c61p-29f));
        } else {
            using A = simd_abi_type_t<T>;
            return dx::broadcast<A, float>(0x1.62e42fefa39efp-1f);
        }
    }

    friend fallback_impl<log1p_t>;

    template <canonical_vector T>
    requires same_as<simd_element_type_t<T>, float> ||
        same_as<simd_element_type_t<T>, double>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr T DPL_VECTORCALL log(T val) noexcept {
        constexpr auto impl = [](T val) {
            auto const [fr, exp] =
                dx::to_tuple_like(dx::frexp(val, frexp_options::reduced));

            auto const one = dx::broadcast<T>(dx::one);
            auto const x =
                mx::pair_ref(fr - mx::single(one)) / (mx::single(one) + fr);
            auto const x2 =
                dx::multiply(dx::get_element<0>(x), dx::get_element<0>(x));
            auto const t = solve_poly(x2);

            auto const ln2 = ln2_val<T>();
            auto s = mx::pair_ref{ln2} * exp;
            s = mx::pair_ref{s} + mx::scale(x, 2);
            return mx::pair_ref{s} +
                dx::multiply(dx::multiply(dx::get_element<0>(x), x2), t);
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
    static constexpr T DPL_VECTORCALL log(T val) noexcept {
        using A = simd_abi_type_t<T>;
        constexpr auto logf = [](rebind_simd_t<T, float> val) {
            auto const [fr, exp] =
                dx::to_tuple_like(dx::frexp(val, frexp_options::reduced));
            auto const vone = dx::broadcast<A, float>(dx::one);
            auto const x =
                dx::divide(dx::subtract(fr, vone), dx::add(fr, vone));
            auto const x2 = dx::multiply(x, x);
            auto const t = solve_poly(x2);

            auto const ln2 = ln2_val<T>();
            auto s = dx::mulacc(dx::multiply(2, x), ln2, exp);
            return dx::mulacc(s, dx::multiply(x2, x), t);
        };

        auto const cval = dx::min(dx::abs(val), dx::max_value);
        auto result = mx::apply_widened<float>(logf, cval);

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
#if DPL_HAS_CONSTEXPR_BUILTIN(__builtin_log)
        if consteval {
            if constexpr (fixed_width_abi<A>) {
                return internal::transform<T>(
                    [](E val) {
                        if constexpr (is_same_v<E, double>) {
                            return __builtin_log(val);
                        } else {
                            return __builtin_logf(static_cast<float>(val));
                        }
                    },
                    val);
            }
        }
#else
        return fallback_impl::log(val);
#endif
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
inline constexpr internal::log_t log{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
