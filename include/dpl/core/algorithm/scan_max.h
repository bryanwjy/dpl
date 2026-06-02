// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"
#include "dpl/core/algorithm/internal/scan.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/operations/compare/max.h"
#  include "dpl/core/type_traits/abi_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct scan_max_t;

struct exscan_max_t;

template <typename T, typename I>
concept unqualified_canonical_exscan_max = requires(T val, I init) {
    {
        exscan_max(internal::abi<T>, val, init)
    } -> canonical_arithmetic_result<T>;
};

template <typename T, typename I>
concept unqualified_extended_exscan_max = requires(T val, I init) {
    { exscan_max(val, init) } -> extended_arithmetic_result<T>;
};

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<M, T>>
concept unqualified_canonical_mexscan_max =
    requires(S src, M mask, T val, I init) {
        {
            exscan_max(internal::abi<A>, src, mask, val, init)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<M, T>>
concept unqualified_extended_mexscan_max =
    requires(S src, M mask, T val, I init) {
        {
            exscan_max(src, mask, val, init)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imexscan_max =
    requires(S src, M mask, T val, I init) {
        {
            exscan_max(internal::abi<A>, src,
                internal::to_const_mask<A, exscan_max_t, S, T>(mask), val, init)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imexscan_max =
    requires(S src, M mask, T val, I init) {
        {
            exscan_max(src,
                internal::to_const_mask<A, exscan_max_t, S, T>(mask), val, init)
        } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
    };

struct exscan_max_t : private scan_base {

    template <typename S, typename M, typename T, typename I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(S src, M mask, T val, I init) noexcept {
        return dx::select(dx::lane_index<S>() < dx::popcount(mask),
            exscan_max_t::operator()(
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
                dx::broadcast<S>(init), dx::max),
            src);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_lane_type_t<T>, typename M::abi_type>;

public:
    template <canonical_vector T, broadcastable_to<T> I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I init) noexcept {
        if constexpr (unqualified_canonical_exscan_max<T, I>) {
            if consteval {
                return scan_base::exclusive(
                    val, dx::broadcast<T>(init), dx::max);
            } else {
                return exscan_max(internal::abi<T>, val, init);
            }
        } else {
            return scan_base::exclusive(val, dx::broadcast<T>(init), dx::max);
        }
    }

    template <extended_vector T, broadcastable_to<T> I>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I init) noexcept {
        if constexpr (unqualified_extended_exscan_max<T, I>) {
            return exscan_max(val, init);
        } else {
            return scan_base::exclusive(val, dx::broadcast<T>(init), dx::max);
        }
    }

    template <canonical_vector S, canonical_mask M, canonical_vector T,
        broadcastable_to<S> I>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init) noexcept {
        using A = abi_type_t<S>;
        if constexpr (same_as<abi_type_t<S>, abi_type_t<M>> &&
            same_as<abi_type_t<T>, abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mexscan_max<S, M, T, I>) {
                if consteval {
                    return exscan_max_t::fallback(src, mask, val, init);
                } else {
                    return exscan_max(internal::abi<A>, src, mask, val, init);
                }
            } else {
                return exscan_max_t::fallback(src, mask, val, init);
            }
        } else if constexpr (unqualified_canonical_mexscan_max<S, M, T, I>) {
            return exscan_max(internal::abi<A>, src, mask, val, init);
        } else {
            return exscan_max_t::fallback(src, mask, val, init);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T, broadcastable_to<S> I>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init) noexcept {
        if constexpr (unqualified_extended_mexscan_max<S, M, T, I>) {
            return exscan_max(src, mask, val, init);
        } else {
            return exscan_max_t::fallback(src, mask, val, init);
        }
    }

    template <canonical_mask M, canonical_vector T,
        broadcastable_to<broadcast_type<M, T>> I>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init) noexcept {
        using A = abi_type_t<M>;
        using S = broadcast_type<M, T>;
        if constexpr (same_as<abi_type_t<T>, abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mexscan_max<zero_t, M, T, I>) {
                if consteval {
                    return operator()(
                        dx::broadcast<S>(dx::zero), mask, val, init);
                } else {
                    return exscan_max(
                        internal::abi<A>, dx::zero, mask, val, init);
                }
            } else {
                return operator()(dx::broadcast<S>(dx::zero), mask, val, init);
            }
        } else if constexpr (unqualified_canonical_mexscan_max<zero_t, M, T,
                                 I>) {
            return exscan_max(internal::abi<A>, dx::zero, mask, val, init);
        } else {
            return operator()(dx::broadcast<S>(dx::zero), mask, val, init);
        }
    }

    template <simd_mask M, simd_vector T,
        broadcastable_to<broadcast_type<M, T>> I>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init) noexcept {
        if constexpr (unqualified_extended_mexscan_max<zero_t, M, T, I>) {
            return exscan_max(dx::zero, mask, val, init);
        } else {
            using S = broadcast_type<M, T>;
            return operator()(dx::broadcast<S>(dx::zero), mask, val, init);
        }
    }

    template <simd_mask M, simd_vector T,
        broadcastable_to<broadcast_type<M, T>> I>
    requires requires(
        M mask, T val, I init) { exscan_max_t::operator()(mask, val, init); }
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
        using A = abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<abi_type_t<T>, abi_type_t<S>>) {
            if constexpr (unqualified_canonical_imexscan_max<S, M, T, I>) {
                if consteval {
                    return exscan_max_t::fallbacki(src, cmask, val, init);
                } else {
                    return exscan_max(internal::abi<A>, src, cmask, val, init);
                }
            } else {
                return exscan_max_t::fallbacki(src, cmask, val, init);
            }
        } else if constexpr (unqualified_canonical_imexscan_max<S, M, T, I>) {
            return exscan_max(internal::abi<A>, src, cmask, val, init);
        } else {
            return exscan_max_t::fallbacki(src, cmask, val, init);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        broadcastable_to<S> I>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imexscan_max<S, M, T, I>) {
            return exscan_max(src, cmask, val, init);
        } else {
            return exscan_max_t::fallback(src, cmask, val, init);
        }
    }

    template <typename M, canonical_vector T, broadcastable_to<T> I>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init) noexcept {
        using A = abi_type_t<T>;
        if constexpr (unqualified_canonical_imexscan_max<zero_t, M, T, I>) {
            if consteval {
                return operator()(dx::broadcast<T>(dx::zero), mask, val, init);
            } else {
                constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
                return exscan_max(internal::abi<A>, dx::zero, cmask, val, init);
            }
        } else {
            return operator()(dx::broadcast<T>(dx::zero), mask, val, init);
        }
    }

    template <typename M, extended_vector T, broadcastable_to<T> I>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init) noexcept {
        if constexpr (unqualified_extended_imexscan_max<zero_t, M, T, I>) {
            return exscan_max(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val, init);
        } else {
            return operator()(dx::broadcast<T>(dx::zero), mask, val, init);
        }
    }

    template <typename M, extended_vector T, broadcastable_to<T> I>
    requires const_mask_for<M, T> && requires(M mask, T val, I init) {
        exscan_max_t::operator()(mask, val, init);
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(
        dx::zero_t, M mask, T val, I init) noexcept {
        return operator()(mask, val, init);
    }
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_canonical_mscan_max = requires(S src, M mask, T val) {
    {
        scan_max(internal::abi<A>, src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<M, T>>
concept unqualified_extended_mscan_max = requires(S src, M mask, T val) {
    {
        scan_max(src, mask, val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<S, T>>
concept unqualified_canonical_imscan_max = requires(S src, T val) {
    {
        scan_max(internal::abi<A>, src, internal::select_mask<M, S, T>(), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename A = common_abi_t<S, T>>
concept unqualified_extended_imscan_max = requires(S src, T val) {
    {
        scan_max(src, internal::select_mask<M, S, T>(), val)
    } -> equivalent_simd_as<canonical_if_zero_t<S, T, A>>;
};

struct scan_max_t : private scan_base {

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(S src, M mask, T val) noexcept {
        return dx::select(dx::lane_index<S>() < dx::popcount(mask),
            scan_max_t::operator()(dx::compress(mask, val)), src);
    }

    template <typename S, typename M, typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(S src, M cmask, T val) noexcept {
        constexpr auto N = dx::popcount(cmask);
        return dx::select(cmask,
            scan_base::inclusive(imm<N>, dx::compress(cmask, val), dx::max),
            src);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_lane_type_t<T>, typename M::abi_type>;

public:
    template <simd_abi A, simd_element_for<A> E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<E, A> operator()(
        basic_vector<E, A> val) noexcept {
        if constexpr (unqualified_canonical_scan_sum<basic_vector<E, A>>) {
            if consteval {
                return scan_base::inclusive(val, dx::max);
            } else {
                return scan_sum(internal::abi<A>, val);
            }
        } else {
            return scan_base::inclusive(val, dx::max);
        }
    }

    template <extended_vector T>
    requires extended_scan_sum<T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val) noexcept {
        if constexpr (unqualified_extended_scan_sum<T>) {
            return scan_sum(val);
        } else {
            return scan_base::inclusive(val, dx::max);
        }
    }

    template <canonical_vector S, canonical_mask M, canonical_vector T>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = abi_type_t<S>;
        if constexpr (same_as<abi_type_t<S>, abi_type_t<M>> &&
            same_as<abi_type_t<T>, abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mscan_max<S, M, T>) {
                if consteval {
                    return scan_max_t::fallback(src, mask, val);
                } else {
                    return scan_max(internal::abi<A>, src, mask, val);
                }
            } else {
                return scan_max_t::fallback(src, mask, val);
            }
        } else if constexpr (unqualified_canonical_mscan_max<S, M, T>) {
            return scan_max(internal::abi<A>, src, mask, val);
        } else {
            return scan_max_t::fallback(src, mask, val);
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        if constexpr (unqualified_extended_mscan_max<S, M, T>) {
            return scan_max(src, mask, val);
        } else {
            return scan_max_t::fallback(src, mask, val);
        }
    }

    template <canonical_mask M, canonical_vector T>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = abi_type_t<M>;
        using S = broadcast_type<M, T>;
        if constexpr (same_as<abi_type_t<T>, abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mscan_max<zero_t, M, T>) {
                if consteval {
                    return operator()(dx::broadcast<S>(dx::zero), mask, val);
                } else {
                    return scan_max(internal::abi<A>, dx::zero, mask, val);
                }
            } else {
                return operator()(dx::broadcast<S>(dx::zero), mask, val);
            }
        } else if constexpr (unqualified_canonical_mscan_max<zero_t, M, T>) {
            return scan_max(internal::abi<A>, dx::zero, mask, val);
        } else {

            return operator()(dx::broadcast<S>(dx::zero), mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_mscan_max<zero_t, M, T>) {
            return scan_max(dx::zero, mask, val);
        } else {
            using S = broadcast_type<M, T>;
            return operator()(dx::broadcast<S>(dx::zero), mask, val);
        }
    }

    template <simd_mask M, simd_vector T>
    requires requires(M mask, T val) { scan_max_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }

    template <canonical_vector S, const_mask_for<S> M, canonical_vector T>
    requires imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        using A = abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<abi_type_t<T>, abi_type_t<S>>) {
            if constexpr (unqualified_canonical_imscan_max<S, M, T>) {
                if consteval {
                    return scan_max_t::fallbacki(src, cmask, val);
                } else {
                    return scan_max(internal::abi<A>, src, cmask, val);
                }
            } else {
                return scan_max_t::fallbacki(src, cmask, val);
            }
        } else if constexpr (unqualified_canonical_imscan_max<S, M, T>) {
            return scan_max(internal::abi<A>, src, cmask, val);
        } else {
            return scan_max_t::fallbacki(src, cmask, val);
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val) noexcept {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imscan_max<S, M, T>) {
            return scan_max(src, cmask, val);
        } else {
            return scan_max_t::fallbacki(src, cmask, val);
        }
    }

    template <typename M, canonical_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        using A = abi_type_t<T>;
        if constexpr (unqualified_canonical_imscan_max<zero_t, M, T>) {
            if consteval {
                return operator()(dx::broadcast<T>(dx::zero), mask, val);
            } else {
                constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
                return scan_max(internal::abi<A>, dx::zero, cmask, val);
            }
        } else {
            return operator()(dx::broadcast<T>(dx::zero), mask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val) noexcept {
        if constexpr (unqualified_extended_imscan_max<zero_t, M, T>) {
            return scan_max(
                dx::zero, dx::to_compatible_const_mask<T>(mask), val);
        } else {
            return operator()(dx::broadcast<T>(dx::zero), mask, val);
        }
    }

    template <typename M, extended_vector T>
    requires const_mask_for<M, T> &&
        requires(M mask, T val) { scan_max_t::operator()(mask, val); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val) noexcept {
        return operator()(mask, val);
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::scan_max_t scan_max{};
DPL_EXPORT inline constexpr internal::exscan_max_t exscan_max{};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
