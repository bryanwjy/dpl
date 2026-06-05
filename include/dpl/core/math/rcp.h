// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/fixup.h"
#include "dpl/core/math/fma.h"
#include "dpl/core/math/internal/floating_point_simd.h"
#include "dpl/core/math/internal/masked_op.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/operations/arithmetic.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_traits.h"
#  include "dpl/core/utility/fpfix.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void rcp(...) noexcept = delete;

struct rcp_t;

template <typename T>
concept unqualified_canonical_rcp = requires(T val) {
    {
        rcp(internal::abi<T>, val)
    } -> canonical_arithmetic_result<T, T, typename T::abi_type>;
};

template <typename T>
concept unqualified_extended_rcp = requires(T val) {
    { rcp(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename T>
concept expression_rcp =
    simd_expression<T> && invocable<rcp_t, simd_expression_result_t<T>>;

template <typename T>
concept decayable_rcp =
    decayable_vector_for<T, operation_category::lane_agnostic> &&
    regular_invocable<rcp_t, canonical_type_t<T>>;

template <typename T>
concept extended_rcp =
    unqualified_extended_rcp<T> || expression_rcp<T> || decayable_rcp<T>;

struct rcp_t : private mx::masked_operation<rcp_t> {
private:
    friend mx::masked_operation<rcp_t>;

    template <floating_point E>
    static constexpr auto useed = []() {
        if constexpr (same_as<E, double>) {
            return unsigned_representation_t<E>(0x7FDE6238DA3C2118);
        } else if constexpr (same_as<E, float>) {
            return unsigned_representation_t<E>(0x7EF311C3);
        } else if constexpr (brain_float<E>) {
            return unsigned_representation_t<E>(0x7EF3);
        } else {
            static_assert(sizeof(E) == 2);
            return unsigned_representation_t<E>(0x7800);
        }
    }();

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto refine(
        basic_vector<E, A> y, basic_vector<E, A> x) noexcept {
        return y * dx::fnmadd(x, y, 2.0);
    }

    template <floating_point E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<float, A> approximate(
        basic_vector<float, A> val) noexcept {
        auto const seed =
            useed<E> - dx::reinterpret<unsigned_representation_t<E>>(val);
        return refine(dx::reinterpret<E>(seed), val);
    }

    template <simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        basic_vector<float, A> val) noexcept {
        auto const result = approximate(val);
        // nan is implicitly handled
        return dx::fixup(val, result,
            fpfix::condition<fpfix::infinity, dx::zero> |
                fpfix::condition<fpfix::zero, fpfix::signed_inf>);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::canonical_masked_math_operator<rcp_t, S, M, T> &&
        requires(
            S src, M mask, T val) { rcp(internal::abi<T>, src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return rcp(internal::abi<T>, src, mask, val);
    }

    template <simd_vector S, typename M, simd_vector T>
    requires mx::extended_masked_math_operator<rcp_t, S, M, T> &&
        requires(S src, M mask, T val) { rcp(src, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(S src, M mask, T val) noexcept {
        return rcp(src, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::canonical_masked_math_zoperator<rcp_t, M, T> &&
        requires(M mask, T val) { rcp(internal::abi<T>, dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return rcp(internal::abi<T>, dx::zero, mask, val);
    }

    template <typename M, simd_vector T>
    requires mx::extended_masked_math_zoperator<rcp_t, M, T> &&
        requires(M mask, T val) { rcp(dx::zero, mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL masked(M mask, T val) noexcept {
        return rcp(dx::zero, mask, val);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    requires floating_point<E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_rcp<basic_vector<E, A>>) {
            if consteval {
                return fallback(val);
            } else {
                return rcp(internal::abi<A>, val);
            }
        } else {
            return fallback(val);
        }
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (!floating_point<E>) &&
        unqualified_canonical_rcp<basic_vector<E, A>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        return rcp(internal::abi<A>, val);
    }

    template <extended_vector T>
    requires extended_rcp<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_rcp<T>) {
            return rcp(val);
        } else if constexpr (expression_rcp<T>) {
            return operator()(dx::evaluate(val));
        } else {
            return operator()(dx::to_canonical(val));
        }
    }

    using mx::masked_operation<rcp_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::rcp_t rcp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
