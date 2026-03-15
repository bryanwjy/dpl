// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/operations/minmax.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void clamp(...) noexcept = delete;

template <typename T, typename L, typename M, typename R>
concept clamp_result = common_order_simd_with<T, common_order_simd_t<L, M, R>>;

template <typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept unqualified_clamp = requires(L val, M low, R high) {
    { clamp(internal::abi<A>, val, low, high) } -> clamp_result<L, M, R>;
};

struct clamp_t {
private:
    template <basic_simd_element E, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd<E, A> val,
        basic_simd<E, A> low, basic_simd<E, A> high) noexcept {
        return dx::min(high, dx::max(val, low));
    }

public:
    template <simd_type L, common_order_simd_with<L> M,
        common_order_simd_with<common_order_simd_t<L, M>> R>
    requires same_abi_simd_as<L, M> && same_abi_simd_as<L, R> &&
        same_abi_simd_as<M, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, M low, R high) noexcept {
        using A = typename L::abi_type;
        if constexpr (unqualified_clamp<L, M, R, A>) {
            if constexpr (basic_simd_type<L> && basic_simd_type<M> &&
                basic_simd_type<R>) {
                if consteval {
                    return fallback(val, low, high);
                } else {
                    return clamp(internal::abi<A>, val, low, high);
                }
            } else {
                return clamp(internal::abi<A>, val, low, high);
            }
        } else if constexpr (basic_simd_type<L> && basic_simd_type<M> &&
            basic_simd_type<R>) {
            return fallback(val, low, high);
        } else {
            return operator()(dx::to_basic_type(val), dx::to_basic_type(low),
                dx::to_basic_type(high));
        }
    }

    template <simd_type L, common_order_simd_with<L> M,
        common_order_simd_with<common_order_simd_t<L, M>> R>
    requires (!same_abi_simd_as<L, M> || !same_abi_simd_as<L, R> ||
                 !same_abi_simd_as<M, R>) &&
        (unqualified_clamp<L, M, R> ||
            unqualified_clamp<basic_type_t<L>, basic_type_t<M>,
                basic_type_t<R>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(L val, M low, R high) noexcept {
        using A = common_abi_t<L, M, R>;
        if constexpr (unqualified_clamp<L, M, R>) {
            return clamp(internal::abi<A>, val, low, high);
        } else {
            return clamp(internal::abi<A>, dx::to_basic_type(val),
                dx::to_basic_type(low), dx::to_basic_type(high));
        }
    }

    template <simd_type TA, broadcastable_to<TA> TB, broadcastable_to<TA> TC>
    requires regular_invocable<clamp_t, TA, TA, TA>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        if constexpr (requires {
                          {
                              clamp(internal::abi<TA>, a, b, c)
                          } -> clamp_result<TA, TA, TA>;
                      }) {
            if constexpr (basic_simd_type<TA>) {
                if consteval {
                    return operator()(
                        a, dx::broadcast<TA>(b), dx::broadcast<TA>(c));
                } else {
                    return clamp(internal::abi<TA>, a, b, c);
                }
            } else {
                return clamp(internal::abi<TA>, a, b, c);
            }
        } else if constexpr (basic_simd_type<TA>) {
            return operator()(a, dx::broadcast<TA>(b), dx::broadcast<TA>(c));
        } else {
            return operator()(dx::to_basic_type(a), b, c);
        }
    }

    template <simd_type TB, broadcastable_to<TB> TA, broadcastable_to<TB> TC>
    requires regular_invocable<clamp_t, TB, TB, TB>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        if constexpr (requires {
                          {
                              clamp(internal::abi<TB>, a, b, c)
                          } -> clamp_result<TB, TB, TB>;
                      }) {
            if constexpr (basic_simd_type<TB>) {
                if consteval {
                    return operator()(
                        a, dx::broadcast<TB>(b), dx::broadcast<TB>(c));
                } else {
                    return clamp(internal::abi<TB>, a, b, c);
                }
            } else {
                return clamp(internal::abi<TB>, a, b, c);
            }
        } else if constexpr (basic_simd_type<TB>) {
            return operator()(a, dx::broadcast<TB>(b), dx::broadcast<TB>(c));
        } else {
            return operator()(dx::to_basic_type(a), b, c);
        }
    }

    template <simd_type TB, broadcastable_to<TB> TA, broadcastable_to<TB> TC>
    requires regular_invocable<clamp_t, TC, TC, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        if constexpr (requires {
                          {
                              clamp(internal::abi<TC>, a, b, c)
                          } -> clamp_result<TC, TC, TC>;
                      }) {
            if constexpr (basic_simd_type<TC>) {
                if consteval {
                    return operator()(
                        a, dx::broadcast<TC>(b), dx::broadcast<TC>(c));
                } else {
                    return clamp(internal::abi<TC>, a, b, c);
                }
            } else {
                return clamp(internal::abi<TC>, a, b, c);
            }
        } else if constexpr (basic_simd_type<TC>) {
            return operator()(a, dx::broadcast<TC>(b), dx::broadcast<TC>(c));
        } else {
            return operator()(dx::to_basic_type(a), b, c);
        }
    }

    template <simd_type TA, common_order_simd_with<TA> TB,
        broadcastable_to<common_order_simd_t<TA, TB>> TC>
    requires regular_invocable<clamp_t, TA, TB, common_order_simd_t<TA, TB>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TB>;
        using TC2 = common_order_simd_t<TA, TB>;
        if constexpr (requires {
                          {
                              clamp(internal::abi<A>, a, b, c)
                          } -> clamp_result<TA, TB, TC2>;
                      }) {
            if constexpr (basic_simd_type<TA> && basic_simd_type<TB>) {
                if consteval {
                    return operator()(a, b, dx::broadcast<TC2>(c));
                } else {
                    return clamp(internal::abi<A>, a, b, c);
                }
            } else {
                return clamp(internal::abi<A>, a, b, c);
            }
        } else if constexpr (basic_simd_type<TA> && basic_simd_type<TB>) {
            return operator()(a, b, dx::broadcast<TC2>(c));
        } else {
            return operator()(dx::to_basic_type(a), dx::to_basic_type(b), c);
        }
    }

    template <simd_type TA, common_order_simd_with<TA> TC,
        broadcastable_to<common_order_simd_t<TA, TC>> TB>
    requires regular_invocable<clamp_t, TA, common_order_simd_t<TA, TC>, TB>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TA, TC>;
        using TB2 = common_order_simd_t<TA, TC>;
        if constexpr (requires {
                          {
                              clamp(internal::abi<A>, a, b, c)
                          } -> clamp_result<TA, TB2, TC>;
                      }) {
            if constexpr (basic_simd_type<TA> && basic_simd_type<TC>) {
                if consteval {
                    return operator()(a, dx::broadcast<TB2>(b), c);
                } else {
                    return clamp(internal::abi<A>, a, b, c);
                }
            } else {
                return clamp(internal::abi<A>, a, b, c);
            }
        } else if constexpr (basic_simd_type<TA> && basic_simd_type<TC>) {
            return operator()(a, dx::broadcast<TB2>(b), c);
        } else {
            return operator()(dx::to_basic_type(a), dx::to_basic_type(b), c);
        }
    }

    template <simd_type TB, common_order_simd_with<TB> TC,
        broadcastable_to<common_order_simd_t<TB, TC>> TA>
    requires regular_invocable<clamp_t, common_order_simd_t<TB, TC>, TB, TC>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(TA a, TB b, TC c) noexcept {
        using A = common_abi_t<TB, TC>;
        using TA2 = common_order_simd_t<TB, TC>;
        if constexpr (requires {
                          {
                              clamp(internal::abi<A>, a, b, c)
                          } -> clamp_result<TA2, TB, TC>;
                      }) {
            if constexpr (basic_simd_type<TB> && basic_simd_type<TC>) {
                if consteval {
                    return operator()(dx::broadcast<TA2>(a), b, c);
                } else {
                    return clamp(internal::abi<A>, a, b, c);
                }
            } else {
                return clamp(internal::abi<A>, a, b, c);
            }
        } else if constexpr (basic_simd_type<TB> && basic_simd_type<TC>) {
            return operator()(dx::broadcast<TA2>(a), b, c);
        } else {
            return operator()(a, dx::to_basic_type(b), dx::to_basic_type(c));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
DPL_EXPORT inline constexpr internal::clamp_t clamp{};
}

DPL_DEFAULT_NAMESPACE_END
