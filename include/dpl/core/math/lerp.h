// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/floating_point_simd.h"
#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/math/fma.h"
#  include "dpl/core/operations/arithmetic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void lerp(...) noexcept = delete;
template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_lerp = requires(L a, M b, R c) {
    { lerp(internal::abi<A>, a, b, c) } -> fma_result<L, M, R>;
};

struct lerp_t : fma_base<lerp_t> {
private:
    friend fma_base<lerp_t>;

    template <simd_abi A, typename TA, typename TB, typename TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto native(A abi, TA start, TB end, TC scale) noexcept
    requires requires {
        {
            lerp(internal::abi<A>, start, end, scale)
        } -> floating_point_simd_with_abi<A>;
    }
    {
        return lerp(internal::abi<A>, start, end, scale);
    }

    template <typename E, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST NODISCARD) static constexpr auto DPL_VECTORCALL
    fallback(basic_vector<E, A> start, basic_vector<E, A> end,
        basic_vector<E, A> scale) noexcept {
        return dx::fmadd(scale, dx::subtract(end, start), start);
    }

public:
    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires same_abi_simd_as<TA, TB> && same_abi_simd_as<TA, TC> &&
        same_abi_simd_as<TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = typename TA::abi_type;
        if constexpr (unqualified_lerp<TA, TB, TC, A>) {
            if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
                canonical_vector<TC>) {
                if consteval {
                    return fallback(a, b, c);
                } else {
                    return lerp(internal::abi<A>, a, b, c);
                }
            } else {
                return lerp(internal::abi<A>, a, b, c);
            }
        } else if constexpr (canonical_vector<TA> && canonical_vector<TB> &&
            canonical_vector<TC>) {
            return fallback(a, b, c);
        } else {
            return fallback(
                dx::to_canonical(a), dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    template <floating_point_simd TA, common_float_simd_with<TA> TB,
        common_float_simd_with<common_float_simd_t<TA, TB>> TC>
    requires (!same_abi_simd_as<TA, TB> || !same_abi_simd_as<TA, TC> ||
                 !same_abi_simd_as<TB, TC>) &&
        (unqualified_lerp<TA, TB, TC> ||
            unqualified_lerp<canonical_type_t<TA>, canonical_type_t<TB>,
                canonical_type_t<TC>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB, TC>;
        if constexpr (unqualified_lerp<TA, TB, TC>) {
            return lerp(internal::abi<A>, a, b, c);
        } else {
            return lerp(internal::abi<A>, dx::to_canonical(a),
                dx::to_canonical(b), dx::to_canonical(c));
        }
    }

    using fma_base<lerp_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::lerp_t lerp{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
