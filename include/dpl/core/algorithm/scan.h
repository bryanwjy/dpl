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
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/type_traits/is_invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename F, typename T>
concept scan_operator_for = simd_vector<T> && regular_invocable<F, T, T> &&
    equivalent_simd_as<T, invoke_result_t<F, T, T>>;

struct scan_t;

struct exscan_t;

template <typename T, typename I, typename BinaryOp>
concept unqualified_canonical_exscan =
    requires(T val, I init, BinaryOp && (*op)()) {
        {
            exscan(internal::abi<T>, val, init, op())
        } -> canonical_arithmetic_result<T>;
    };

template <typename T, typename I, typename BinaryOp>
concept unqualified_extended_exscan =
    requires(T val, I init, BinaryOp && (*op)()) {
        { exscan(val, init, op()) } -> extended_arithmetic_result<T>;
    };

template <typename S, typename M, typename T, typename I, typename BinaryOp,
    typename A = common_abi_t<M, T>>
concept unqualified_canonical_mexscan =
    requires(S src, M mask, T val, I init, BinaryOp && (*op)()) {
        {
            exscan(internal::abi<A>, src, mask, val, init, op())
        } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I, typename BinaryOp,
    typename A = common_abi_t<M, T>>
concept unqualified_extended_mexscan =
    requires(S src, M mask, T val, I init, BinaryOp && (*op)()) {
        {
            exscan(src, mask, val, init, op())
        } -> extended_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename I, typename BinaryOp,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imexscan = requires(
    S src, M mask, T val, I init, BinaryOp && (*op)()) {
    {
        exscan(internal::abi<A>, src,
            internal::to_const_mask<A, exscan_t, S, T>(mask), val, init, op())
    } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
};

template <typename S, typename M, typename T, typename I, typename BinaryOp,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imexscan =
    requires(S src, M mask, T val, I init, BinaryOp && (*op)()) {
        {
            exscan(src, internal::to_const_mask<A, exscan_t, S, T>(mask), val,
                init, op())
        } -> extended_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

struct exscan_t : private scan_base {
    template <typename S, typename M, typename T, typename I, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(
        S src, M mask, T val, I init, BinaryOp&& op) noexcept {
        return dx::select(dx::lane_index<S>() < dx::popcount(mask),
            scan_base::exclusive(dx::popcount(mask), dx::compress(mask, val),
                dx::broadcast<canonical_type_t<T>>(init),
                __DPL forward<BinaryOp>(op)),
            src);
    }

    template <typename S, typename M, typename T, typename I, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(
        S src, M cmask, T val, I init, BinaryOp&& op) noexcept {
        constexpr auto V = (1zu << dx::popcount(cmask)) - 1;
        constexpr auto N = dx::popcount(cmask);
        return dx::selecti<V>(
            scan_base::exclusive(imm<N>, dx::compress(cmask, val),
                dx::broadcast<canonical_type_t<S>>(init),
                __DPL forward<BinaryOp>(op)),
            src);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_lane_type_t<T>, typename M::abi_type>;

public:
    template <canonical_vector T, broadcastable_to<T> I,
        scan_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I init, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_canonical_exscan<T, I, Op>) {
            if consteval {
                return scan_base::exclusive(val,
                    dx::broadcast<canonical_type_t<T>>(init),
                    __DPL forward<Op>(op));
            } else {
                return exscan(
                    internal::abi<T>, val, init, __DPL forward<Op>(op));
            }
        } else {
            return scan_base::exclusive(val,
                dx::broadcast<canonical_type_t<T>>(init),
                __DPL forward<Op>(op));
        }
    }

    template <extended_vector T, broadcastable_to<canonical_type_t<T>> I,
        scan_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, I init, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_exscan<T, I, Op>) {
            return exscan(val, init, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<T>) {
            return operator()(dx::evaluate(val), init, __DPL forward<Op>(op));
        } else {
            return scan_base::exclusive(val,
                dx::broadcast<canonical_type_t<T>>(init),
                __DPL forward<Op>(op));
        }
    }

    template <canonical_vector S, canonical_mask M, canonical_vector T,
        broadcastable_to<S> I, scan_operator_for<T> Op>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mexscan<S, M, T, I, Op>) {
                if consteval {
                    return exscan_t::fallback(
                        src, mask, val, init, __DPL forward<Op>(op));
                } else {
                    return exscan(internal::abi<A>, src, mask, val, init,
                        __DPL forward<Op>(op));
                }
            } else {
                return exscan_t::fallback(
                    src, mask, val, init, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_mexscan<S, M, T, I, Op>) {
            return exscan(
                internal::abi<A>, src, mask, val, init, __DPL forward<Op>(op));
        } else {
            return exscan_t::fallback(
                src, mask, val, init, __DPL forward<Op>(op));
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T, broadcastable_to<S> I,
        scan_operator_for<T> Op>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_mexscan<S, M, T, I, Op>) {
            return exscan(src, mask, val, init, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<S> || simd_expression<M> ||
            simd_expression<T>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(val), init, __DPL forward<Op>(op));
        } else {
            return exscan_t::fallback(
                src, mask, val, init, __DPL forward<Op>(op));
        }
    }

    template <canonical_mask M, canonical_vector T,
        broadcastable_to<broadcast_type<M, T>> I, scan_operator_for<T> Op>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<M>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mexscan<zero_t, M, T, I, Op>) {
                if consteval {
                    return exscan_t::fallback(
                        dx::zero, mask, val, init, __DPL forward<Op>(op));
                } else {
                    return exscan(internal::abi<A>, dx::zero, mask, val, init,
                        __DPL forward<Op>(op));
                }
            } else {
                return exscan_t::fallback(
                    dx::zero, mask, val, init, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_mexscan<zero_t, M, T, I,
                                 Op>) {
            return exscan(internal::abi<A>, dx::zero, mask, val, init,
                __DPL forward<Op>(op));
        } else {
            return exscan_t::fallback(
                dx::zero, mask, val, init, __DPL forward<Op>(op));
        }
    }

    template <simd_mask M, simd_vector T,
        broadcastable_to<broadcast_type<M, T>> I, scan_operator_for<T> Op>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_mexscan<zero_t, M, T, I, Op>) {
            return exscan(dx::zero, mask, val, init, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(dx::evaluate(mask), dx::evaluate(val), init,
                __DPL forward<Op>(op));
        } else {
            return exscan_t::fallback(
                dx::zero, mask, val, init, __DPL forward<Op>(op));
        }
    }

    template <simd_mask M, simd_vector T,
        broadcastable_to<broadcast_type<M, T>> I, scan_operator_for<T> Op>
    requires invocable<exscan_t, M, T, I, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val, I init,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        return operator()(mask, val, init, __DPL forward<Op>(op));
    }

    template <canonical_vector S, const_mask_for<S> M, canonical_vector T,
        broadcastable_to<S> I, scan_operator_for<T> Op>
    requires imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<S>>) {
            if constexpr (unqualified_canonical_imexscan<S, M, T, I, Op>) {
                if consteval {
                    return exscan_t::fallbacki(
                        src, cmask, val, init, __DPL forward<Op>(op));
                } else {
                    return exscan(internal::abi<A>, src, cmask, val, init,
                        __DPL forward<Op>(op));
                }
            } else {
                return exscan_t::fallbacki(
                    src, cmask, val, init, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_imexscan<S, M, T, I, Op>) {
            return exscan(internal::abi<A>, src, cmask, val, init,
                __DPL forward<Op>(op));
        } else {
            return exscan_t::fallbacki(
                src, cmask, val, init, __DPL forward<Op>(op));
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        broadcastable_to<S> I, scan_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, I init,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imexscan<S, M, T, I, Op>) {
            return exscan(src, cmask, val, init, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<S> || simd_expression<T>) {
            return operator()(dx::evaluate(src), cmask, dx::evaluate(val), init,
                __DPL forward<Op>(op));
        } else {
            return exscan_t::fallbacki(
                src, cmask, val, init, __DPL forward<Op>(op));
        }
    }

    template <typename M, canonical_vector T, broadcastable_to<T> I,
        scan_operator_for<T> Op>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<T>;
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_canonical_imexscan<zero_t, M, T, I, Op>) {
            if consteval {
                return exscan_t::fallbacki(
                    dx::zero, mask, val, init, __DPL forward<Op>(op));
            } else {
                return exscan(internal::abi<A>, dx::zero, cmask, val, init,
                    __DPL forward<Op>(op));
            }
        } else {
            return exscan_t::fallbacki(
                dx::zero, mask, val, init, __DPL forward<Op>(op));
        }
    }

    template <typename M, extended_vector T,
        broadcastable_to<canonical_type_t<T>> I, scan_operator_for<T> Op>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, I init, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_imexscan<zero_t, M, T, I, Op>) {
            return exscan(dx::zero, dx::to_compatible_const_mask<T>(mask), val,
                init, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<T>) {
            return operator()(
                mask, dx::evaluate(val), init, __DPL forward<Op>(op));
        } else {
            return exscan_t::fallbacki(
                dx::zero, mask, val, init, __DPL forward<Op>(op));
        }
    }

    template <typename M, simd_vector T, broadcastable_to<T> I,
        scan_operator_for<T> Op>
    requires const_mask_for<M, T> && invocable<exscan_t, M, T, I, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val, I init,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        return operator()(mask, val, init, __DPL forward<Op>(op));
    }
};

template <typename T, typename BinaryOp>
concept unqualified_canonical_scan = requires(T val, BinaryOp && (*op)()) {
    { scan(internal::abi<T>, val, op()) } -> canonical_arithmetic_result<T>;
};

template <typename T, typename BinaryOp>
concept unqualified_extended_scan = requires(T val, BinaryOp && (*op)()) {
    { scan(val, op()) } -> extended_arithmetic_result<T>;
};

template <typename S, typename M, typename T, typename BinaryOp,
    typename A = common_abi_t<M, T>>
concept unqualified_canonical_mscan =
    requires(S src, M mask, T val, BinaryOp && (*op)()) {
        {
            scan(internal::abi<A>, src, mask, val, op())
        } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename BinaryOp,
    typename A = common_abi_t<M, T>>
concept unqualified_extended_mscan =
    requires(S src, M mask, T val, BinaryOp && (*op)()) {
        {
            scan(src, mask, val, op())
        } -> extended_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename BinaryOp,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imscan =
    requires(S src, M mask, T val, BinaryOp && (*op)()) {
        {
            scan(internal::abi<A>, src,
                internal::to_const_mask<A, scan_t, S, T>(mask), val, op())
        } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename BinaryOp,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imscan =
    requires(S src, M mask, T val, BinaryOp && (*op)()) {
        {
            scan(src, internal::to_const_mask<A, scan_t, S, T>(mask), val, op())
        } -> extended_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

struct scan_t : private scan_base {
    template <typename S, typename M, typename T, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallback(
        S src, M mask, T val, BinaryOp&& op) noexcept {
        return dx::select(dx::lane_index<S>() < dx::popcount(mask),
            scan_base::inclusive(dx::popcount(mask), dx::compress(mask, val),
                __DPL forward<BinaryOp>(op)),
            src);
    }

    template <typename S, typename M, typename T, typename BinaryOp>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto fallbacki(
        S src, M cmask, T val, BinaryOp&& op) noexcept {
        constexpr auto V = (1zu << dx::popcount(cmask)) - 1;
        constexpr auto N = dx::popcount(cmask);
        return dx::selecti<V>(
            scan_base::inclusive(
                imm<N>, dx::compress(cmask, val), __DPL forward<BinaryOp>(op)),
            src);
    }

    template <typename M, typename T>
    using broadcast_type DPL_NODEBUG =
        rebind_simd_t<T, simd_lane_type_t<T>, typename M::abi_type>;

public:
    template <canonical_vector T, scan_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_canonical_scan<T, Op>) {
            if consteval {
                return scan_base::inclusive(val, __DPL forward<Op>(op));
            } else {
                return scan(internal::abi<T>, val, __DPL forward<Op>(op));
            }
        } else {
            return scan_base::inclusive(val, __DPL forward<Op>(op));
        }
    }

    template <extended_vector T, scan_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_scan<T, Op>) {
            return scan(val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<T>) {
            return operator()(dx::evaluate(val), __DPL forward<Op>(op));
        } else {
            return scan_base::inclusive(val, __DPL forward<Op>(op));
        }
    }

    template <canonical_vector S, canonical_mask M, canonical_vector T,
        broadcastable_to<S>, scan_operator_for<T> Op>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mscan<S, M, T, Op>) {
                if consteval {
                    return scan_t::fallback(
                        src, mask, val, __DPL forward<Op>(op));
                } else {
                    return scan(internal::abi<A>, src, mask, val,
                        __DPL forward<Op>(op));
                }
            } else {
                return scan_t::fallback(src, mask, val, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_mscan<S, M, T, Op>) {
            return scan(
                internal::abi<A>, src, mask, val, __DPL forward<Op>(op));
        } else {
            return scan_t::fallback(src, mask, val, __DPL forward<Op>(op));
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T,
        scan_operator_for<T> Op>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_mscan<S, M, T, Op>) {
            return scan(src, mask, val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<S> || simd_expression<M> ||
            simd_expression<T>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(val), __DPL forward<Op>(op));
        } else {
            return scan_t::fallback(src, mask, val, __DPL forward<Op>(op));
        }
    }

    template <canonical_mask M, canonical_vector T, scan_operator_for<T> Op>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<M>;
        using S = broadcast_type<M, T>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mscan<zero_t, M, T, Op>) {
                if consteval {
                    return scan_t::fallback(
                        dx::zero, mask, val, __DPL forward<Op>(op));
                } else {
                    return scan(internal::abi<A>, dx::zero, mask, val,
                        __DPL forward<Op>(op));
                }
            } else {
                return scan_t::fallback(
                    dx::zero, mask, val, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_mscan<zero_t, M, T, Op>) {
            return scan(
                internal::abi<A>, dx::zero, mask, val, __DPL forward<Op>(op));
        } else {
            return scan_t::fallback(
                dx::zero, mask, val, __DPL forward<Op>(op));
        }
    }

    template <simd_mask M, simd_vector T, scan_operator_for<T> Op>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_mscan<zero_t, M, T, Op>) {
            return scan(dx::zero, mask, val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(val), __DPL forward<Op>(op));
        } else {
            return scan_t::fallback(
                dx::zero, mask, val, __DPL forward<Op>(op));
        }
    }

    template <simd_mask M, simd_vector T, scan_operator_for<T> Op>
    requires invocable<scan_t, M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        return operator()(mask, val, __DPL forward<Op>(op));
    }

    template <canonical_vector S, const_mask_for<S> M, canonical_vector T,
        scan_operator_for<T> Op>
    requires imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<S>>) {
            if constexpr (unqualified_canonical_imscan<S, M, T, Op>) {
                if consteval {
                    return scan_t::fallbacki(
                        src, cmask, val, __DPL forward<Op>(op));
                } else {
                    return scan(internal::abi<A>, src, cmask, val,
                        __DPL forward<Op>(op));
                }
            } else {
                return scan_t::fallbacki(
                    src, cmask, val, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_imscan<S, M, T, Op>) {
            return scan(
                internal::abi<A>, src, cmask, val, __DPL forward<Op>(op));
        } else {
            return scan_t::fallbacki(src, cmask, val, __DPL forward<Op>(op));
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        scan_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imscan<S, M, T, Op>) {
            return scan(src, cmask, val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<S> || simd_expression<T>) {
            return operator()(dx::evaluate(src), cmask, dx::evaluate(val),
                __DPL forward<Op>(op));
        } else {
            return scan_t::fallbacki(src, cmask, val, __DPL forward<Op>(op));
        }
    }

    template <typename M, canonical_vector T, scan_operator_for<T> Op>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<T>;
        if constexpr (unqualified_canonical_imscan<zero_t, M, T, Op>) {
            if consteval {
                return scan_t::fallbacki(
                    dx::zero, mask, val, __DPL forward<Op>(op));
            } else {
                constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
                return scan(internal::abi<A>, dx::zero, cmask, val,
                    __DPL forward<Op>(op));
            }
        } else {
            return scan_t::fallbacki(
                dx::zero, mask, val, __DPL forward<Op>(op));
        }
    }

    template <typename M, extended_vector T, scan_operator_for<T> Op>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_imscan<zero_t, M, T, Op>) {
            return scan(dx::zero, dx::to_compatible_const_mask<T>(mask), val,
                __DPL forward<Op>(op));
        } else if constexpr (simd_expression<T>) {
            return operator()(mask, dx::evaluate(val), __DPL forward<Op>(op));
        } else {
            return scan_t::fallbacki(
                dx::zero, mask, val, __DPL forward<Op>(op));
        }
    }

    template <typename M, simd_vector T, scan_operator_for<T> Op>
    requires const_mask_for<M, T> && invocable<scan_t, M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        return operator()(mask, val, __DPL forward<Op>(op));
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::scan_t scan{};
DPL_EXPORT inline constexpr internal::exscan_t exscan{};
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
