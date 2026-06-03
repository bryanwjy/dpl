// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/reduce.h"

#if !DPL_MODULES
#  include "dpl/core/operations/arithmetic/add.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

void hsum(...) noexcept = delete;

template <typename T>
concept unqualified_canonical_hsum = requires(T val) {
    { hsum(internal::abi<T>, val) } -> canonical_arithmetic_result<T>;
};

template <typename T>
concept unqualified_extended_hsum = requires(T val) {
    { hsum(val) } -> extended_arithmetic_result<T>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mhsum = requires(S src, M mask, T val) {
    {
        hsum(internal::abi<A>, src, mask, val)
    } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mhsum = requires(S src, M mask, T val) {
    {
        hsum(src, mask, val)
    } -> extended_arithmetic_result<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<S, T>>
concept unqualified_canonical_imhsum = requires(S src, T val) {
    {
        hsum(internal::abi<A>, src, internal::select_mask<M, S, T>(), val)
    } -> extended_arithmetic_result<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<S, T>>
concept unqualified_extended_imhsum = requires(S src, T val) {
    {
        hsum(src, internal::select_mask<M, S, T>(), val)
    } -> extended_arithmetic_result<canonical_if_zero_t<S, T, A>>;
};

struct hsum_t : private reduction_base {
private:
    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(S src, M mask, T val) noexcept {
        return dx::select(dx::lane_index<S>() < dx::popcount(mask),
            hsum_t::operator()(dx::compress(mask, val)), src);
    }

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(S src, M cmask, T val) noexcept {
        constexpr auto N = dx::popcount(cmask);
        return dx::select(cmask,
            reduction_base::execute(imm<N>, dx::compress(cmask, val), dx::add),
            src);
    }

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_hsum<basic_vector<E, A>>) {
            if consteval {
                return reduction_base::execute(val, dx::add);
            } else {
                return hsum(internal::abi<A>, val);
            }
        } else {
            return reduction_base::execute(val, dx::add);
        }
    }

    template <extended_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_hsum<T>) {
            return hsum(val);
        } else {
            return reduction_base::execute(val, dx::add);
        }
    }

    template <canonical_vector S, canonical_mask M, canonical_vector T>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mhsum<S, M, T>) {
                if consteval {
                    return hsum_t::fallback(src, mask, val);
                } else {
                    return hsum(internal::abi<A>, src, mask, val);
                }
            } else {
                return hsum_t::fallback(src, mask, val);
            }
        } else if constexpr (unqualified_canonical_mhsum<S, M, T>) {
            return hsum(internal::abi<A>, src, mask, val);
        } else {
            return hsum_t::fallback(src, mask, val);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mhsum<S, M, T>) {
            return hsum(src, mask, val);
        } else {
            return hsum_t::fallback(src, mask, val);
        }
    }

    template <canonical_mask M, canonical_vector T>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<M>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mhsum<zero_t, M, T>) {
                if consteval {
                    return hsum_t::fallback(dx::zero, mask, val);
                } else {
                    return hsum(internal::abi<A>, dx::zero, mask, val);
                }
            } else {
                return hsum_t::fallback(dx::zero, mask, val);
            }
        } else if constexpr (unqualified_canonical_mhsum<zero_t, M, T>) {
            return hsum(internal::abi<A>, dx::zero, mask, val);
        } else {

            return hsum_t::fallback(dx::zero, mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mhsum<zero_t, M, T>) {
            return hsum(dx::zero, mask, val);
        } else {
            return hsum_t::fallback(dx::zero, mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires requires(M mask, T val) { hsum_t::operator()(mask, val); }
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
            if constexpr (unqualified_canonical_imhsum<S, M, T>) {
                if consteval {
                    return hsum_t::fallbacki(src, cmask, val);
                } else {
                    return hsum(internal::abi<A>, src, cmask, val);
                }
            } else {
                return hsum_t::fallbacki(src, cmask, val);
            }
        } else if constexpr (unqualified_canonical_imhsum<S, M, T>) {
            return hsum(internal::abi<A>, src, cmask, val);
        } else {
            return hsum_t::fallbacki(src, cmask, val);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imhsum<S, M, T>) {
            return hsum(src, cmask, val);
        } else {
            return hsum_t::fallbacki(src, cmask, val);
        }
    }

    template <typename M, canonical_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<T>;
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_canonical_imhsum<zero_t, M, T>) {
            if consteval {
                return hsum_t::fallbacki(dx::zero, cmask, val);
            } else {
                return hsum(internal::abi<A>, dx::zero, cmask, val);
            }
        } else {
            return hsum_t::fallbacki(dx::zero, cmask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_extended_imhsum<zero_t, M, T>) {
            return hsum(dx::zero, cmask, val);
        } else {
            return hsum_t::fallbacki(dx::zero, cmask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T> &&
        requires(M mask, T val) { hsum_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::hsum_t hsum{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
