// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/operations/compare/min.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void hmin(...) noexcept = delete;

template <typename T>
concept unqualified_canonical_hmin = requires(T val) {
    { hmin(internal::abi<T>, val) } -> canonical_arithmetic_result<T>;
};

template <typename T>
concept unqualified_extended_hmin = requires(T val) {
    { hmin(val) } -> vector_with_common_abi<typename T::abi_type>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mhmin = requires(S src, M mask, T val) {
    {
        hmin(internal::abi<A>, src, mask, val)
    } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mhmin = requires(S src, M mask, T val) {
    { hmin(src, mask, val) } -> vector_with_common_abi<A>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<S, T>>
concept unqualified_canonical_imhmin = requires(S src, T val) {
    {
        hmin(internal::abi<A>, src, internal::select_mask<M, S, T>(), val)
    } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<S, T>>
concept unqualified_extended_imhmin = requires(S src, T val) {
    {
        hmin(src, internal::select_mask<M, S, T>(), val)
    } -> vector_with_common_abi<A>;
};

struct hmin_t : private reduction_base {
private:
    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(S src, M mask, T val) noexcept {
        return dx::select(dx::lane_index<S>() < dx::popcount(mask),
            hmin_t::operator()(dx::compress(mask, val)), src);
    }

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(S src, M cmask, T val) noexcept {
        constexpr auto N = dx::popcount(cmask);
        return dx::select(cmask,
            reduction_base::execute(imm<N>, dx::compress(cmask, val), dx::min),
            src);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_hmin<basic_vector<E, A>>) {
            if consteval {
                return reduction_base::execute(val, dx::max);
            } else {
                return hmin(internal::abi<A>, val);
            }
        } else {
            return reduction_base::execute(val, dx::max);
        }
    }

    template <extended_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_hmin<T>) {
            return hmin(val);
        } else {
            return reduction_base::execute(val, dx::max);
        }
    }

    template <canonical_vector S, canonical_mask M, canonical_vector T>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mhmin<S, M, T>) {
                if consteval {
                    return hmin_t::fallback(src, mask, val);
                } else {
                    return hmin(internal::abi<A>, src, mask, val);
                }
            } else {
                return hmin_t::fallback(src, mask, val);
            }
        } else if constexpr (unqualified_canonical_mhmin<S, M, T>) {
            return hmin(internal::abi<A>, src, mask, val);
        } else {
            return hmin_t::fallback(src, mask, val);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mhmin<S, M, T>) {
            return hmin(src, mask, val);
        } else if constexpr (simd_expression<S> || simd_expression<M> ||
            simd_expression<T>) {
            return operator()(
                dx::evaluate(src), dx::evaluate(mask), dx::evaluate(val));
        } else {
            return hmin_t::fallback(src, mask, val);
        }
    }

    template <canonical_mask M, canonical_vector T>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<M>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mhmin<zero_t, M, T>) {
                if consteval {
                    return hmin_t::fallback(dx::zero, mask, val);
                } else {
                    return hmin(internal::abi<A>, dx::zero, mask, val);
                }
            } else {
                return hmin_t::fallback(dx::zero, mask, val);
            }
        } else if constexpr (unqualified_canonical_mhmin<zero_t, M, T>) {
            return hmin(internal::abi<A>, dx::zero, mask, val);
        } else {

            return hmin_t::fallback(dx::zero, mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mhmin<zero_t, M, T>) {
            return hmin(dx::zero, mask, val);
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val));
        } else {
            return hmin_t::fallback(dx::zero, mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires invocable<hmin_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <canonical_vector S, const_mask_for<S> M, canonical_vector T>
    requires imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<S>>) {
            if constexpr (unqualified_canonical_imhmin<S, M, T>) {
                if consteval {
                    return hmin_t::fallbacki(src, cmask, val);
                } else {
                    return hmin(internal::abi<A>, src, cmask, val);
                }
            } else {
                return hmin_t::fallbacki(src, cmask, val);
            }
        } else if constexpr (unqualified_canonical_imhmin<S, M, T>) {
            return hmin(internal::abi<A>, src, cmask, val);
        } else {
            return hmin_t::fallbacki(src, cmask, val);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imhmin<S, M, T>) {
            return hmin(src, cmask, val);
        } else if constexpr (simd_expression<S> || simd_expression<T>) {
            return operator()(dx::evaluate(src), cmask, dx::evaluate(val));
        } else {
            return hmin_t::fallbacki(src, cmask, val);
        }
    }

    template <typename M, canonical_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<T>;
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_canonical_imhmin<zero_t, M, T>) {
            if consteval {
                return hmin_t::fallbacki(dx::zero, cmask, val);
            } else {
                return hmin(internal::abi<A>, dx::zero, cmask, val);
            }
        } else {
            return hmin_t::fallbacki(dx::zero, cmask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_extended_imhmin<zero_t, M, T>) {
            return hmin(dx::zero, cmask, val);
        } else if constexpr (simd_expression<T>) {
            return operator()(cmask, dx::evaluate(val));
        } else {
            return hmin_t::fallbacki(dx::zero, cmask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T> && invocable<hmin_t, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::hmin_t hmin{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
