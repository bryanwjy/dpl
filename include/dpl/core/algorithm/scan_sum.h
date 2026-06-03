// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/internal/scan.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct scan_sum_t;

struct exscan_sum_t;

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<M, T>>
concept unqualified_canonical_mexscan_sum =
    requires(S src, M mask, T val, I init) {
        {
            exscan_sum(internal::abi<A>, src, mask, val, init)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<M, T>>
concept unqualified_extended_mexscan_sum =
    requires(S src, M mask, T val, I init) {
        {
            exscan_sum(src, mask, val, init)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imexscan_sum =
    requires(S src, M mask, T val, I init) {
        {
            exscan_sum(internal::abi<A>, src,
                internal::to_const_mask<A, exscan_sum_t, S, T>(mask), val, init)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imexscan_sum =
    requires(S src, M mask, T val, I init) {
        {
            exscan_sum(src,
                internal::to_const_mask<A, exscan_sum_t, S, T>(mask), val, init)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

struct exscan_sum_t : private exscan_sum_base {

    template <typename S, typename M, typename T, typename I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(S src, M mask, T val, I init) noexcept {
        return dx::select(dx::lane_index<S>() < dx::popcount(mask),
            exscan_sum_base::operator()(
                dx::compress(mask, val), dx::broadcast<T>(init)),
            src);
    }

    template <typename S, typename M, typename T, typename I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(S src, M cmask, T val, I init) noexcept {
        constexpr auto V = (1zu << dx::popcount(cmask)) - 1;
        constexpr auto N = dx::popcount(cmask);
        return dx::selecti<V>(
            scan_base::exclusive(imm<N>, dx::compress(cmask, val),
                dx::broadcast<S>(init), dx::add),
            src);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_lane_type_t<T>, typename M::abi_type>;

public:
    using exscan_sum_base::operator();

    template <canonical_vector S, canonical_mask M, canonical_vector T,
        broadcastable_to<S> I>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init) noexcept {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mexscan_sum<S, M, T, I>) {
                if consteval {
                    return exscan_sum_t::fallback(src, mask, val, init);
                } else {
                    return exscan_sum(internal::abi<A>, src, mask, val, init);
                }
            } else {
                return exscan_sum_t::fallback(src, mask, val, init);
            }
        } else if constexpr (unqualified_canonical_mexscan_sum<S, M, T, I>) {
            return exscan_sum(internal::abi<A>, src, mask, val, init);
        } else {
            return exscan_sum_t::fallback(src, mask, val, init);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T, broadcastable_to<S> I>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init) noexcept {
        if constexpr (unqualified_extended_mexscan_sum<S, M, T, I>) {
            return exscan_sum(src, mask, val, init);
        } else {
            return exscan_sum_t::fallback(src, mask, val, init);
        }
    }

    template <canonical_mask M, canonical_vector T,
        broadcastable_to<broadcast_type<M, T>> I>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init) noexcept {
        using A = simd_abi_type_t<M>;
        using S = broadcast_type<M, T>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mexscan_sum<zero_t, M, T, I>) {
                if consteval {
                    return exscan_sum_t::fallback(dx::zero, mask, val, init);
                } else {
                    return exscan_sum(
                        internal::abi<A>, dx::zero, mask, val, init);
                }
            } else {
                return exscan_sum_t::fallback(dx::zero, mask, val, init);
            }
        } else if constexpr (unqualified_canonical_mexscan_sum<zero_t, M, T,
                                 I>) {
            return exscan_sum(internal::abi<A>, dx::zero, mask, val, init);
        } else {
            return exscan_sum_t::fallback(dx::zero, mask, val, init);
        }
    }

    template <simd_mask M, simd_vector T,
        broadcastable_to<broadcast_type<M, T>> I>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init) noexcept {
        if constexpr (unqualified_extended_mexscan_sum<zero_t, M, T, I>) {
            return exscan_sum(dx::zero, mask, val, init);
        } else {
            using S = broadcast_type<M, T>;
            return exscan_sum_t::fallback(dx::zero, mask, val, init);
        }
    }

    template <simd_mask M, simd_vector T,
        broadcastable_to<broadcast_type<M, T>> I>
    requires requires(
        M mask, T val, I init) { exscan_sum_t::operator()(mask, val, init); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T val, I init) noexcept {
        return operator()(mask, val, init);
    }

    template <canonical_vector S, const_mask_for<S> M, canonical_vector T,
        broadcastable_to<S> I>
    requires imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init) noexcept {
        using A = simd_abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<S>>) {
            if constexpr (unqualified_canonical_imexscan_sum<S, M, T, I>) {
                if consteval {
                    return exscan_sum_t::fallbacki(src, cmask, val, init);
                } else {
                    return exscan_sum(internal::abi<A>, src, cmask, val, init);
                }
            } else {
                return exscan_sum_t::fallbacki(src, cmask, val, init);
            }
        } else if constexpr (unqualified_canonical_imexscan_sum<S, M, T, I>) {
            return exscan_sum(internal::abi<A>, src, cmask, val, init);
        } else {
            return exscan_sum_t::fallbacki(src, cmask, val, init);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        broadcastable_to<S> I>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imexscan_sum<S, M, T, I>) {
            return exscan_sum(src, cmask, val, init);
        } else {
            return exscan_sum_t::fallbacki(src, cmask, val, init);
        }
    }

    template <typename M, canonical_vector T, broadcastable_to<T> I>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init) noexcept {
        using A = simd_abi_type_t<T>;
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_canonical_imexscan_sum<zero_t, M, T, I>) {
            if consteval {
                return exscan_sum_t::fallbacki(dx::zero, mask, val, init);
            } else {

                return exscan_sum(internal::abi<A>, dx::zero, cmask, val, init);
            }
        } else {
            return exscan_sum_t::fallbacki(dx::zero, mask, val, init);
        }
    }

    template <typename M, extended_vector T, broadcastable_to<T> I>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_extended_imexscan_sum<zero_t, M, T, I>) {
            return exscan_sum(dx::zero, cmask, val, init);
        } else {
            return exscan_sum_t::fallbacki(dx::zero, cmask, val, init);
        }
    }

    template <typename M, extended_vector T, broadcastable_to<T> I>
    requires const_mask_for<M, T> && requires(M mask, T val, I init) {
        exscan_sum_t::operator()(mask, val, init);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T val, I init) noexcept {
        return operator()(mask, val, init);
    }
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mscan_sum = requires(S src, M mask, T val) {
    {
        scan_sum(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mscan_sum = requires(S src, M mask, T val) {
    {
        scan_sum(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_canonical_imscan_sum = requires(S src, R val) {
    {
        scan_sum(internal::abi<A>, src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_as<S>;
};

template <typename S, typename M, typename R, typename A = common_abi_t<S, R>>
concept unqualified_extended_imscan_sum = requires(S src, R val) {
    {
        scan_sum(src, internal::select_mask<M, S, R>(), val)
    } -> equivalent_simd_as<S>;
};

struct scan_sum_t : private scan_sum_base {

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(S src, M mask, T val) noexcept {
        return dx::select(dx::lane_index<S>() < dx::popcount(mask),
            scan_sum_base::operator()(dx::compress(mask, val)), src);
    }

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(S src, M cmask, T val) noexcept {
        constexpr auto N = dx::popcount(cmask);
        return dx::select(cmask,
            scan_base::inclusive(imm<N>, dx::compress(cmask, val), dx::add),
            src);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_lane_type_t<T>, typename M::abi_type>;

public:
    using scan_sum_base::operator();

    template <canonical_vector S, canonical_mask M, canonical_vector T>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mscan_sum<S, M, T>) {
                if consteval {
                    return scan_sum_t::fallback(src, mask, val);
                } else {
                    return scan_sum(internal::abi<A>, src, mask, val);
                }
            } else {
                return scan_sum_t::fallback(src, mask, val);
            }
        } else if constexpr (unqualified_canonical_mscan_sum<S, M, T>) {
            return scan_sum(internal::abi<A>, src, mask, val);
        } else {
            return scan_sum_t::fallback(src, mask, val);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mscan_sum<S, M, T>) {
            return scan_sum(src, mask, val);
        } else {
            return scan_sum_t::fallback(src, mask, val);
        }
    }

    template <canonical_mask M, canonical_vector T>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<M>;
        using S = broadcast_type<M, T>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mscan_sum<zero_t, M, T>) {
                if consteval {
                    return scan_sum_t::fallback(dx::zero, mask, val);
                } else {
                    return scan_sum(internal::abi<A>, dx::zero, mask, val);
                }
            } else {
                return scan_sum_t::fallback(dx::zero, mask, val);
            }
        } else if constexpr (unqualified_canonical_mscan_sum<zero_t, M, T>) {
            return scan_sum(internal::abi<A>, dx::zero, mask, val);
        } else {

            return scan_sum_t::fallback(dx::zero, mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mscan_sum<zero_t, M, T>) {
            return scan_sum(dx::zero, mask, val);
        } else {
            using S = broadcast_type<M, T>;
            return operator()(dx::broadcast<S>(dx::zero), mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires requires(M mask, T val) { scan_sum_t::operator()(mask, val); }
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
            if constexpr (unqualified_canonical_imscan_sum<S, M, T>) {
                if consteval {
                    return scan_sum_t::fallbacki(src, cmask, val);
                } else {
                    return scan_sum(internal::abi<A>, src, cmask, val);
                }
            } else {
                return scan_sum_t::fallbacki(src, cmask, val);
            }
        } else if constexpr (unqualified_canonical_imscan_sum<S, M, T>) {
            return scan_sum(internal::abi<A>, src, cmask, val);
        } else {
            return scan_sum_t::fallbacki(src, cmask, val);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imscan_sum<S, M, T>) {
            return scan_sum(src, cmask, val);
        } else {
            return scan_sum_t::fallbacki(src, cmask, val);
        }
    }

    template <typename M, canonical_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = simd_abi_type_t<T>;
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_canonical_imscan_sum<zero_t, M, T>) {
            if consteval {
                return scan_sum_t::fallbacki(dx::zero, cmask, val);
            } else {
                return scan_sum(internal::abi<A>, dx::zero, cmask, val);
            }
        } else {
            return scan_sum_t::fallbacki(dx::zero, cmask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_imscan_sum<zero_t, M, T>) {
            return scan_sum(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else {
            return scan_sum_t::fallbacki(dx::zero, mask, val);
        }
    }

    template <typename M, simd_vector T>
    requires const_mask_for<M, T> &&
        requires(M mask, T val) { scan_sum_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {

DPL_EXPORT inline constexpr internal::scan_sum_t scan_sum{};

DPL_EXPORT inline constexpr internal::exscan_sum_t exscan_sum{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
