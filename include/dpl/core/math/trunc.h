// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/compliance.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/ilogb.h"
#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/constants/one.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/select.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/utility/rounding.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct trunc_t;

template <typename T>
concept unqualified_canonical_trunc = requires(T val) {
    {
        round(internal::abi<T>, val, rounding::to_pos_inf)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_trunc = requires(T val) {
    {
        round(val, rounding::to_pos_inf)
    } -> extended_operation_vector<typename T::abi_type>;
};

template <typename T>
concept expression_trunc = simd_expression<T> &&
    regular_invocable<trunc_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_trunc =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<trunc_t, canonical_type_t<T>>;

template <typename T>
concept extended_trunc =
    unqualified_extended_trunc<T> || expression_trunc<T> || decayable_trunc<T>;

template <typename T>
concept unqualified_canonical_truncne = requires(T val) {
    {
        round(internal::abi<T>, val, rounding::to_pos_inf | rounding::no_exc)
    } -> equivalent_simd_as<T>;
};

template <typename T>
concept unqualified_extended_truncne = requires(T val) {
    {
        round(val, rounding::to_pos_inf | rounding::no_exc)
    } -> extended_operation_vector<typename T::abi_type>;
};

template <typename T>
concept expression_truncne = simd_expression<T> &&
    regular_invocable<trunc_t, simd_expression_result_t<T>, rounding::no_exc_t>;

template <typename T>
concept decayable_truncne =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<trunc_t, canonical_type_t<T>, rounding::no_exc_t>;

template <typename T>
concept extended_truncne = unqualified_extended_trunc<T> ||
    expression_truncne<T> || decayable_truncne<T>;

struct trunc_t : mx::masked_operation<trunc_t> {
private:
    friend mx::masked_operation<trunc_t>;
    static constexpr auto noexc = rounding::to_zero | rounding::no_exc;
    static constexpr auto exc = rounding::to_zero;

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<E, A> val) noexcept {
        // Based on musl libm
        using sint = signed_representation_t<E>;
        using uint = unsigned_representation_t<E>;
        static constexpr auto width =
            dx::broadcast<sint, A>(sizeof(E) * char_bit_v);
        static constexpr auto margin = width - dx::mantissa_width_v<E>;
        auto const exp = [](auto exp) {
            return dx::select(exp < margin, dx::one, exp);
        }(mx::ilogb(mx::compliance::unsafe, val) + margin);

        auto const m = dx::broadcast<E, A>(dx::all_bits) >> exp;
        auto const result = dx::bwandnot(val, m);
        return dx::select(exp >= width ||
                dx::bwand(dx::reinterpret<uint>(val),
                    dx::reinterpret<uint>(m)) == dx::zero,
            val, result);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<trunc_t, S, M, T> &&
        requires(S src, M mask, T val) {
            round(internal::abi<T>, src, mask, val, trunc_t::exc);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return round(internal::abi<T>, src, mask, val, trunc_t::exc);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<trunc_t, S, M, T> &&
        requires(S src, M mask, T val) { round(src, mask, val, trunc_t::exc); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return round(src, mask, val, trunc_t::exc);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<trunc_t, M, T> &&
        requires(M mask, T val) {
            round(internal::abi<T>, dx::zero, mask, val, trunc_t::exc);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return round(internal::abi<T>, dx::zero, mask, val, trunc_t::exc);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<trunc_t, M, T> &&
        requires(M mask, T val) { round(dx::zero, mask, val, trunc_t::exc); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return round(dx::zero, mask, val, trunc_t::exc);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<trunc_t, S, M, T,
                 rounding::no_exc_t> &&
        requires(S src, M mask, T val) {
            round(internal::abi<T>, src, mask, val, trunc_t::noexc);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, src, mask, val, trunc_t::noexc);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<trunc_t, S, M, T,
                 rounding::no_exc_t> &&
        requires(
            S src, M mask, T val) { round(src, mask, val, trunc_t::noexc); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        S src, M mask, T val, rounding::no_exc_t) noexcept {
        return round(src, mask, val, trunc_t::noexc);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<trunc_t, M, T,
                 rounding::no_exc_t> &&
        requires(M mask, T val) {
            round(internal::abi<T>, dx::zero, mask, val, trunc_t::noexc);
        }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, rounding::no_exc_t) noexcept {
        return round(internal::abi<T>, dx::zero, mask, val, trunc_t::noexc);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<trunc_t, M, T,
                 rounding::no_exc_t> &&
        requires(M mask, T val) { round(dx::zero, mask, val, trunc_t::noexc); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(
        M mask, T val, rounding::no_exc_t) noexcept {
        return round(dx::zero, mask, val, trunc_t::noexc);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_trunc<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val, trunc_t::exc);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_trunc<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return round(internal::abi<A>, val, trunc_t::exc);
    }

    template <extended_vector T>
    requires extended_trunc<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_trunc<T>) {
            return round(val, trunc_t::exc);
        } else if constexpr (expression_trunc<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_canonical_truncne<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return round(internal::abi<A>, val, trunc_t::noexc);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_truncne<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val, rounding::no_exc_t) noexcept {
        return round(internal::abi<A>, val, trunc_t::noexc);
    }

    template <extended_vector T>
    requires extended_truncne<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, rounding::no_exc_t) noexcept {
        if constexpr (unqualified_extended_trunc<T>) {
            return round(val, trunc_t::noexc);
        } else if constexpr (expression_truncne<T>) {
            return operator()(dx::evaluate(val), rounding::no_exc);
        } else {
            return operator()(dx::to_canonical(val), rounding::no_exc);
        }
    }

    using mx::masked_operation<trunc_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::trunc_t trunc{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
