// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/algorithm/compress.h"

#if !DPL_MODULES
#  include "dpl/core/basic/const_mask.h"
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/operations/bit.h"
#  include "dpl/core/operations/lane_index.h"
#  include "dpl/core/operations/logical.h"
#  include "dpl/core/operations/permute.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

struct reduce_t;

template <typename F, typename T>
concept reduction_operator_for = simd_vector<T> && regular_invocable<F, T, T> &&
    core_convertible_to<invoke_result_t<F, T, T>, T>;

template <typename T, typename BinaryOp>
concept unqualified_canonical_reduce = requires(T val, BinaryOp && (*op)()) {
    { reduce(internal::abi<T>, val, op()) } -> canonical_arithmetic_result<T>;
};

template <typename T, typename BinaryOp>
concept unqualified_extended_reduce = requires(T val, BinaryOp && (*op)()) {
    { reduce(val, op()) } -> extended_arithmetic_result<T>;
};

template <typename S, typename M, typename T, typename BinaryOp,
    typename A = common_abi_t<M, T>>
concept unqualified_canonical_mreduce =
    requires(S src, M mask, T val, BinaryOp && (*op)()) {
        {
            reduce(internal::abi<A>, src, mask, val, op())
        } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename BinaryOp,
    typename A = common_abi_t<M, T>>
concept unqualified_extended_mreduce =
    requires(S src, M mask, T val, BinaryOp && (*op)()) {
        {
            reduce(src, mask, val, op())
        } -> extended_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename BinaryOp,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_canonical_imreduce =
    requires(S src, M mask, T val, BinaryOp && (*op)()) {
        {
            reduce(internal::abi<A>, src,
                internal::to_const_mask<A, reduce_t, S, T>(mask), val, op())
        } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
    };

template <typename S, typename M, typename T, typename BinaryOp,
    typename A = common_abi_t<canonical_if_zero_t<S, T>, T>>
concept unqualified_extended_imreduce = requires(
    S src, M mask, T val, BinaryOp && (*op)()) {
    {
        reduce(src, internal::to_const_mask<A, reduce_t, S, T>(mask), val, op())
    } -> canonical_arithmetic_result<canonical_if_zero_t<S, T, A>>;
};

struct reduction_base {
protected:
    template <integral_constant_like N, fixed_width_vector T,
        reduction_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL execute(N size, T val, Op op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        constexpr auto simd_size = simd_abi_traits<T>::size();
        static_assert(simd_size >= N::value);
        return [&op]<size_t I>(this auto self, auto val, immediate<I> offset) {
            if constexpr (I >= N::value) {
                return val;
            } else {
                auto const idx = []<size_t... Is>(index_sequence<Is...>) {
                    return index_sequence<(Is ^ I)...>{};
                }(iota_sequence<T>);

                return self(
                    __DPL invoke(op, val, dx::permute(val, idx)), imm<I * 2>);
            }
        }(val, imm<1zu>);
    }

    template <simd_vector T, reduction_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL execute(
        size_t size, T val, Op op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        constexpr auto simd_size = simd_abi_traits<T>::size();
        auto const idx = dx::lane_index<T>();
        using idx_t = remove_const_t<decltype(idx)>;
        using sint = typename idx_t::value_type;
        for (auto offset = 1zu; offset < size; offset <<= 1) {
            auto const perm = dx::bwxor(idx, static_cast<sint>(offset));
            val = __DPL invoke(op, val, dx::permute(val, perm));
        }

        return val;
    }

    template <simd_vector T, reduction_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL execute(T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (fixed_width_vector<T>) {
            return reduction_base::execute(
                simd_abi_traits<T>::size, val, __DPL forward<Op>(op));
        } else {
            return reduction_base::execute(
                simd_abi_traits<T>::size(), val, __DPL forward<Op>(op));
        }
    }
};

struct reduce_t : private reduction_base {
private:
    template <typename S, typename M, typename T, typename Op>
    static constexpr auto fallback(S src, M mask, T value, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        auto const compressed = dx::compress(mask, value);
        auto const idx = dx::lane_index<T>();
        auto const count = dx::popcount(mask);
        return dx::select(idx < count,
            reduction_base::execute(
                dx::popcount(mask), compressed, __DPL forward<Op>(op)),
            src);
    }

    template <typename S, typename M, typename T, typename Op>
    static constexpr auto fallbacki(S src, M cmask, T value, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        constexpr auto count = dx::popcount(cmask);
        constexpr auto V = (1zu << count) - 1;

        auto const compressed = dx::compress(cmask, value);
        return dx::selecti<V>(reduction_base::execute(imm<count>, compressed,
                                  __DPL forward<Op>(op)),
            src);
    }

public:
    template <canonical_vector T, reduction_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_canonical_reduce<T, Op>) {
            if consteval {
                return reduction_base::execute(val, __DPL forward<Op>(op));
            } else {
                return reduce(internal::abi<T>, val, __DPL forward<Op>(op));
            }
        } else {
            return reduction_base::execute(val, __DPL forward<Op>(op));
        }
    }

    template <extended_vector T, reduction_operator_for<T> Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_reduce<T, Op>) {
            return reduce(val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<T>) {
            return operator()(dx::evaluate(val), __DPL forward<Op>(op));
        } else {
            return reduction_base::execute(val, __DPL forward<Op>(op));
        }
    }

    template <canonical_vector S, canonical_mask M, canonical_vector T,
        broadcastable_to<S>, reduction_operator_for<T> Op>
    requires maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<S>;
        if constexpr (same_as<simd_abi_type_t<S>, simd_abi_type_t<M>> &&
            same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mreduce<S, M, T, Op>) {
                if consteval {
                    return reduce_t::fallback(
                        src, mask, val, __DPL forward<Op>(op));
                } else {
                    return reduce(internal::abi<A>, src, mask, val,
                        __DPL forward<Op>(op));
                }
            } else {
                return reduce_t::fallback(
                    src, mask, val, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_mreduce<S, M, T, Op>) {
            return reduce(
                internal::abi<A>, src, mask, val, __DPL forward<Op>(op));
        } else {
            return reduce_t::fallback(src, mask, val, __DPL forward<Op>(op));
        }
    }

    template <simd_vector S, simd_mask M, simd_vector T,
        reduction_operator_for<T> Op>
    requires (extended_vector<S> || extended_mask<M> || extended_vector<T>) &&
        maskable_args<S, M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_mreduce<S, M, T, Op>) {
            return reduce(src, mask, val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<S> || simd_expression<M> ||
            simd_expression<T>) {
            return operator()(dx::evaluate(src), dx::evaluate(mask),
                dx::evaluate(val), __DPL forward<Op>(op));
        } else {
            return reduce_t::fallback(src, mask, val, __DPL forward<Op>(op));
        }
    }

    template <canonical_mask M, canonical_vector T,
        reduction_operator_for<T> Op>
    requires zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<M>;
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<M>>) {
            if constexpr (unqualified_canonical_mreduce<zero_t, M, T, Op>) {
                if consteval {
                    return reduce_t::fallback(
                        dx::zero, mask, val, __DPL forward<Op>(op));
                } else {
                    return reduce(internal::abi<A>, dx::zero, mask, val,
                        __DPL forward<Op>(op));
                }
            } else {
                return reduce_t::fallback(
                    dx::zero, mask, val, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_mreduce<zero_t, M, T, Op>) {
            return reduce(
                internal::abi<A>, dx::zero, mask, val, __DPL forward<Op>(op));
        } else {
            return reduce_t::fallback(
                dx::zero, mask, val, __DPL forward<Op>(op));
        }
    }

    template <simd_mask M, simd_vector T, reduction_operator_for<T> Op>
    requires (extended_mask<M> || extended_vector<T>) && zmaskable_args<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        if constexpr (unqualified_extended_mreduce<zero_t, M, T, Op>) {
            return reduce(dx::zero, mask, val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<M> || simd_expression<T>) {
            return operator()(
                dx::evaluate(mask), dx::evaluate(val), __DPL forward<Op>(op));
        } else {
            return reduce_t::fallback(
                dx::zero, mask, val, __DPL forward<Op>(op));
        }
    }

    template <simd_mask M, simd_vector T, reduction_operator_for<T> Op>
    requires invocable<reduce_t, M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        return operator()(mask, val, __DPL forward<Op>(op));
    }

    template <canonical_vector S, const_mask_for<S> M, canonical_vector T,
        reduction_operator_for<T> Op>
    requires imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<S>;
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (same_as<simd_abi_type_t<T>, simd_abi_type_t<S>>) {
            if constexpr (unqualified_canonical_imreduce<S, M, T, Op>) {
                if consteval {
                    return reduce_t::fallbacki(
                        src, cmask, val, __DPL forward<Op>(op));
                } else {
                    return reduce(internal::abi<A>, src, cmask, val,
                        __DPL forward<Op>(op));
                }
            } else {
                return reduce_t::fallbacki(
                    src, cmask, val, __DPL forward<Op>(op));
            }
        } else if constexpr (unqualified_canonical_imreduce<S, M, T, Op>) {
            return reduce(
                internal::abi<A>, src, cmask, val, __DPL forward<Op>(op));
        } else {
            return reduce_t::fallbacki(src, cmask, val, __DPL forward<Op>(op));
        }
    }

    template <simd_vector S, const_mask_for<S> M, simd_vector T,
        reduction_operator_for<T> Op>
    requires (extended_vector<S> || extended_vector<T>) &&
        imm_maskable_args<S, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(S src, M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        constexpr auto cmask = dx::to_compatible_const_mask<S>(mask);
        if constexpr (unqualified_extended_imreduce<S, M, T, Op>) {
            return reduce(src, cmask, val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<T> || simd_expression<S>) {
            return operator()(dx::evaluate(src), cmask, dx::evaluate(val),
                __DPL forward<Op>(op));
        } else {
            return reduce_t::fallback(src, cmask, val, __DPL forward<Op>(op));
        }
    }

    template <typename M, canonical_vector T, reduction_operator_for<T> Op>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        using A = simd_abi_type_t<T>;
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_canonical_imreduce<zero_t, M, T, Op>) {
            if consteval {
                return reduce_t::fallbacki(
                    dx::zero, cmask, val, __DPL forward<Op>(op));
            } else {
                return reduce(internal::abi<A>, dx::zero, cmask, val,
                    __DPL forward<Op>(op));
            }
        } else {
            return reduce_t::fallbacki(
                dx::zero, mask, val, __DPL forward<Op>(op));
        }
    }

    template <typename M, extended_vector T, reduction_operator_for<T> Op>
    requires const_mask_for<M, T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(M mask, T val, Op&& op) noexcept(
        is_nothrow_invocable_v<Op, T, T>) {
        constexpr auto cmask = dx::to_compatible_const_mask<T>(mask);
        if constexpr (unqualified_extended_imreduce<zero_t, M, T, Op>) {
            return reduce(dx::zero, cmask, val, __DPL forward<Op>(op));
        } else if constexpr (simd_expression<T>) {
            return operator()(mask, dx::evaluate(val), __DPL forward<Op>(op));
        } else {
            return reduce_t::fallbacki(
                dx::zero, cmask, val, __DPL forward<Op>(op));
        }
    }

    template <typename M, simd_vector T, reduction_operator_for<T> Op>
    requires const_mask_for<M, T> && invocable<reduce_t, M, T, Op>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(dx::zero_t, M mask, T val,
        Op&& op) noexcept(is_nothrow_invocable_v<Op, T, T>) {
        return operator()(mask, val, __DPL forward<Op>(op));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT inline constexpr internal::reduce_t reduce{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
