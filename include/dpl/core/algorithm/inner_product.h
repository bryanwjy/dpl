// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/hsum.h"

#if !DPL_MODULES
#  include "dpl/core/operations/arithmetic/multiply.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct inner_product_t;
void inner_product(...) noexcept = delete;

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_canonical_inner_product = requires(L lhs, R rhs) {
    {
        inner_product(internal::abi<A>, lhs, rhs)
    } -> canonical_arithmetic_result<L, R, A>;
};

template <typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_extended_inner_product = requires(L lhs, R rhs) {
    { inner_product(lhs, rhs) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_minner_product = requires(
    S src, M mask, L lhs, R rhs) {
    {
        inner_product(internal::abi<A>, src, mask, lhs, rhs)
    } -> canonical_arithmetic_result<
        canonical_if_zero_t<S, operation_result_t<inner_product_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_minner_product =
    requires(S src, M mask, L lhs, R rhs) {
        { inner_product(src, mask, lhs, rhs) } -> vector_with_common_abi<A>;
    };

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_canonical_iminner_product = requires(
    S src, M mask, L lhs, R rhs) {
    {
        inner_product(internal::abi<A>, src,
            internal::to_const_mask<A, inner_product_t, S, L, R>(mask), lhs,
            rhs)
    } -> canonical_arithmetic_result<
        canonical_if_zero_t<S, operation_result_t<inner_product_t, L, R>, A>>;
};

template <typename S, typename M, typename L, typename R,
    typename A = common_abi_t<L, R, M>>
concept unqualified_extended_iminner_product =
    requires(S src, M mask, L lhs, R rhs) {
        {
            inner_product(src,
                internal::to_const_mask<A, inner_product_t, S, L, R>(mask), lhs,
                rhs)
        } -> vector_with_common_abi<A>;
    };

struct inner_product_t : private binary_operation_base<inner_product_t> {
private:
    friend binary_operation_base<inner_product_t>;

    template <simd_vector L, simd_vector R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(L lhs, R rhs) noexcept {
        return dx::hsum(dx::multiply(lhs, rhs));
    }

    template <typename S, typename M, simd_vector L, simd_vector R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(
        S src, M mask, L lhs, R rhs) noexcept {
        return dx::hsum(src, mask, dx::multiply(lhs, rhs));
    }

public:
    template <canonical_vector L, canonical_vector R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_canonical_inner_product<L, R>) {
            if consteval {
                if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<R>>) {
                    return fallback(lhs, rhs);
                } else {
                    return inner_product(
                        internal::abi<common_abi_t<L, R>>, lhs, rhs);
                }
            } else {
                return inner_product(
                    internal::abi<common_abi_t<L, R>>, lhs, rhs);
            }
        } else {
            return fallback(lhs, rhs);
        }
    }

    template <simd_vector L, simd_vector R>
    requires extended_vector<L> || extended_vector<R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_inner_product<L, R>) {
            return inner_product(lhs, rhs);
        } else {
            return fallback(lhs, rhs);
        }
    }

    using binary_operation_base<inner_product_t>::operator();

    template <canonical_vector S, canonical_mask M, canonical_vector L,
        canonical_vector R>
    requires maskable_args<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<L>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<R>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_minner_product<S, M, L, R>) {
                if consteval {
                    return inner_product_t::fallback(src, mask, lhs, rhs);
                } else {
                    return inner_product(internal::abi<A>, src, mask, lhs, rhs);
                }
            } else {
                return inner_product_t::fallback(src, mask, lhs, rhs);
            }
        } else if constexpr (unqualified_canonical_minner_product<S, M, L, R>) {
            return inner_product(internal::abi<A>, src, mask, lhs, rhs);
        } else {
            return inner_product_t::fallback(src, mask, lhs, rhs);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<L> ||
                 extended_vector<R>) &&
        maskable_args<S, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_minner_product<S, M, L, R>) {
            return inner_product(src, mask, lhs, rhs);
        } else {
            return inner_product_t::fallback(src, mask, lhs, rhs);
        }
    }

    template <canonical_mask M, canonical_vector L, canonical_vector R>
    requires zmaskable_args<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using A = simd_abi_type_t<M>;
        if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<R>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_minner_product<zero_t, M, L,
                              R>) {
                if consteval {
                    return inner_product_t::fallback(dx::zero, mask, lhs, rhs);
                } else {
                    return inner_product(
                        internal::abi<A>, dx::zero, mask, lhs, rhs);
                }
            } else {
                return inner_product_t::fallback(dx::zero, mask, lhs, rhs);
            }
        } else if constexpr (unqualified_canonical_minner_product<zero_t, M, L,
                                 R>) {
            return inner_product(internal::abi<A>, dx::zero, mask, lhs, rhs);
        } else {
            return inner_product_t::fallback(dx::zero, mask, lhs, rhs);
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires (extended_mask<M> || extended_vector<L> || extended_vector<R>) &&
        zmaskable_args<M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        if constexpr (unqualified_extended_minner_product<zero_t, M, L, R>) {
            return inner_product(dx::zero, mask, lhs, rhs);
        } else {
            return inner_product_t::fallback(dx::zero, mask, lhs, rhs);
        }
    }

    template <simd_mask M, simd_vector L, simd_vector R>
    requires invocable<inner_product_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }

    template <canonical_vector S, const_mask_for<S> M, canonical_vector L,
        canonical_vector R>
    requires imm_maskable_args<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        using A = simd_abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<simd_abi_type_t<L>, simd_abi_type_t<S>> &&
            same_as<simd_abi_type_t<R>, simd_abi_type_t<S>>) {
            if constexpr (unqualified_canonical_iminner_product<S, M, L, R>) {
                if consteval {
                    return inner_product_t::fallback(src, cmask, lhs, rhs);
                } else {
                    return inner_product(
                        internal::abi<A>, src, cmask, lhs, rhs);
                }
            } else {
                return inner_product_t::fallback(src, cmask, lhs, rhs);
            }
        } else if constexpr (unqualified_canonical_iminner_product<S, M, L,
                                 R>) {
            return inner_product(internal::abi<A>, src, cmask, lhs, rhs);
        } else {
            return inner_product_t::fallback(src, cmask, lhs, rhs);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector L, simd_vector R>
    requires (extended_vector<S> || extended_vector<L> || extended_vector<R>) &&
        imm_maskable_args<S, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, L lhs, R rhs) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_iminner_product<S, M, L, R>) {
            return inner_product(src, cmask, lhs, rhs);
        } else {
            return inner_product_t::fallback(src, cmask, lhs, rhs);
        }
    }

    template <typename M, canonical_vector L, canonical_vector R>
    requires const_mask_for<M, operation_result_t<inner_product_t, L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<inner_product_t, L, R>;
        using A = typename S::abi_type;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_canonical_iminner_product<zero_t, M, L, R>) {
            if consteval {
                return inner_product_t::fallback(dx::zero, cmask, lhs, rhs);
            } else {
                return inner_product(
                    internal::abi<A>, dx::zero, cmask, lhs, rhs);
            }
        } else {
            return inner_product_t::fallback(dx::zero, cmask, lhs, rhs);
        }
    }

    template <typename M, simd_vector L, simd_vector R>
    requires (extended_vector<L> || extended_vector<R>) &&
        const_mask_for<M, operation_result_t<inner_product_t, L, R>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, L lhs, R rhs) noexcept {
        using S = operation_result_t<inner_product_t, L, R>;
        using A = typename S::abi_type;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_iminner_product<zero_t, M, L, R>) {
            return inner_product(dx::zero, cmask, lhs, rhs);
        } else {
            return inner_product_t::fallback(dx::zero, cmask, lhs, rhs);
        }
    }

    template <typename M, simd_vector L, simd_vector R>
    requires const_mask_for<M, operation_result_t<inner_product_t, L, R>> &&
        invocable<inner_product_t, M, L, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, L lhs, R rhs) noexcept {
        return operator()(mask, lhs, rhs);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::inner_product_t inner_product{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
