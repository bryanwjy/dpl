// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/compatible_mask_for.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/core/type_traits/to_simd_type.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/type_traits/declval.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void select(...) noexcept = delete;
template <auto>
void select(...) noexcept = delete;

template <typename L, typename R>
using ternary_result_t DPL_NODEBUG =
    decltype(false ? __DPL declval<L>() : __DPL declval<R>());

template <typename L, typename R>
struct ternary_type {};
template <typename L, typename R>
using ternary_type_t DPL_NODEBUG = typename ternary_type<L, R>::type;

template <simd_type L, simd_element R>
struct ternary_type<L, R> : ternary_type<typename L::value_type, R> {};

template <simd_element L, simd_type R>
struct ternary_type<L, R> : ternary_type<L, typename R::value_type> {};

template <simd_type L, simd_type R>
struct ternary_type<L, R> :
    ternary_type<typename L::value_type, typename R::value_type> {};

template <simd_element L, simd_element R>
requires requires {
    typename ternary_result_t<L, R>;
    requires common_arithmetic_with<ternary_result_t<L, R>, L>;
    requires common_arithmetic_with<ternary_result_t<L, R>, R>;
}
struct ternary_type<L, R> {
    using type DPL_NODEBUG = ternary_result_t<L, R>;
};

template <typename L, typename R>
struct ternary_simd {};
template <typename L, typename R>
using ternary_simd_t DPL_NODEBUG = typename ternary_simd<L, R>::type;

template <simd_type L, simd_type R>
requires requires { typename ternary_type_t<L, R>; }
struct ternary_simd<L, R> {
    using type DPL_NODEBUG =
        basic_simd<ternary_type_t<L, R>, common_abi_t<L, R>>;
};

template <typename L, typename R>
concept selectable_with = common_size_simd_with<R, L> && requires {
    typename ternary_type_t<L, R>;
    requires common_size_with<ternary_type_t<L, R>, common_size_type_t<R, L>>;
};

template <typename C, typename L, typename R>
concept only_unqualified_selectible = !basic_simd_class<L> ||
    !basic_simd_class<R> || !basic_simd_class<R> || !same_abi_simd_as<L, R> ||
    !same_abi_simd_as<C, R> || !same_abi_simd_as<C, L>;

template <typename A, typename C, typename L, typename R>
concept unqualified_select = requires(
    C cond, L lhs, R rhs) { select(internal::abi<A>, cond, lhs, rhs); };

template <typename A, typename M, typename L, typename R>
concept unqualified_selecti =
    immediate_mask_for<M, L> && requires(L lhs, R rhs) {
        bit_select<immediate_mask_v<L, M>>(internal::abi<A>, lhs, rhs);
    };

struct select_t {
private:
    template <simd_element M, simd_element ET, simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd_mask<M, A> mask,
        basic_simd<ET, A> tval, basic_simd<EF, A> fval) noexcept {
        using ER = ternary_type_t<ET, EF>;
        return internal::transform<basic_simd<ER, A>>(
            mask, tval, fval, [](bool cond, ET tval, EF fval) -> ER {
                return cond ? tval : fval;
            });
    }

    template <simd_element M, simd_element ET, simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_simd_mask<M, A> mask,
        basic_simd_mask<ET, A> tval, basic_simd_mask<EF, A> fval) noexcept {
        using ER = common_size_type_t<M, ET, EF>;
        return internal::transform<basic_simd_mask<ER, A>>(
            mask, tval, fval, [](bool cond, bool tval, bool fval) -> bool {
                return cond ? tval : fval;
            });
    }

    template <auto V, simd_element ET, simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL
        fallbacki(basic_simd<ET, A> tval, basic_simd<EF, A> fval) noexcept {
        static constexpr immediate_mask<element_count<ET, A>, V> mask{};
        using ER = ternary_type_t<ET, EF>;
        return internal::itransform<basic_simd<ER, A>>(
            tval, fval, [](size_t idx, ET tval, EF fval) -> ER {
                return mask[idx] ? tval : fval;
            });
    }

    template <auto V, simd_element ET, simd_element EF, simd_abi A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_simd_mask<ET, A> tval, basic_simd_mask<EF, A> fval) noexcept {
        static constexpr immediate_mask<element_count<ET, A>, V> mask{};
        using ER = common_size_type_t<ET, EF>;
        return internal::itransform<basic_simd_mask<ER, A>>(
            tval, fval, [](size_t idx, ET tval, EF fval) -> ER {
                return mask[idx] ? tval : fval;
            });
    }

    template <typename M, typename T, typename F>
    using mask_result DPL_NODEBUG =
        basic_simd_mask<common_size_type_t<T, F>, common_abi_t<M, T, F>>;

    template <typename M, typename T, typename F>
    using result DPL_NODEBUG =
        basic_simd<ternary_type_t<T, F>, common_abi_t<M, T, F>>;

public:
    template <basic_simd_mask_type M, basic_simd_type TT, basic_simd_type TF>
    requires selectable_with<TT, TF> && compatible_mask_for<M, TT> &&
        compatible_mask_for<M, TF> && same_abi_simd_as<TT, TF> &&
        same_abi_simd_as<M, TT> && same_abi_simd_as<M, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr result<M, TT, TF> DPL_VECTORCALL operator()(
        M mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF, M>;
        if constexpr (requires {
                          select(internal::abi<A>, mask, tval, fval);
                      }) {
            using R = decltype(select(internal::abi<A>, mask, tval, fval));
            static_assert(same_as<R, decltype(fallback(mask, tval, fval))>);
            if consteval {
                return fallback(mask, tval, fval);
            } else {
                return select(internal::abi<A>, mask, tval, fval);
            }
        } else {
            return fallback(mask, tval, fval);
        }
    }

    template <simd_mask_type M, simd_type TT, simd_type TF>
    requires selectable_with<TT, TF> && compatible_mask_for<M, TT> &&
        compatible_mask_for<M, TF> && only_unqualified_selectible<M, TT, TF> &&
        unqualified_select<common_abi_t<TT, TF, M>, M, TT, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, TT tval,
        TF fval) noexcept -> equivalent_simd_as<result<M, TT, TF>> auto {
        using A = common_abi_t<TT, TF, M>;
        return select(internal::abi<A>, mask, tval, fval);
    }

    template <simd_mask_type M, simd_type TT, simd_type TF>
    requires selectable_with<TT, TF> && compatible_mask_for<M, TT> &&
        compatible_mask_for<M, TF> &&
        (!basic_simd_mask_type<M> || !basic_simd_type<TT> ||
            !basic_simd_type<TF>) &&
        (!unqualified_select<common_abi_t<TT, TF, M>, M, TT, TF>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, TT tval,
        TF fval) noexcept -> equivalent_simd_as<result<M, TT, TF>> auto {
        return operator()(dx::to_basic_type(mask), dx::to_basic_type(tval),
            dx::to_basic_type(fval));
    }

    template <basic_simd_mask_type M, basic_simd_mask_type TT,
        basic_simd_mask_type TF>
    requires common_size_simd_with<TT, TF> && common_size_simd_with<M, TT> &&
        common_size_simd_with<M, TF> && same_abi_simd_as<TT, TF> &&
        same_abi_simd_as<M, TT> && same_abi_simd_as<M, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr mask_result<M, TT, TF> DPL_VECTORCALL operator()(
        M mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF, M>;
        if constexpr (requires {
                          select(internal::abi<A>, mask, tval, fval);
                      }) {
            if consteval {
                return fallback(mask, tval, fval);
            } else {
                return select(internal::abi<A>, mask, tval, fval);
            }
        } else {
            return fallback(mask, tval, fval);
        }
    }

    template <simd_mask_type M, simd_mask_type TT, simd_mask_type TF>
    requires common_size_simd_with<TT, TF> && common_size_simd_with<M, TT> &&
        common_size_simd_with<M, TF> &&
        only_unqualified_selectible<M, TT, TF> &&
        unqualified_select<common_abi_t<TT, TF, M>, M, TT, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, TT tval, TF fval) noexcept
        -> common_size_simd_with<mask_result<M, TT, TF>> auto {
        using A = common_abi_t<TT, TF, M>;
        return select(internal::abi<A>, mask, tval, fval);
    }

    template <simd_mask_type M, simd_mask_type TT, simd_mask_type TF>
    requires common_size_simd_with<TT, TF> && common_size_simd_with<M, TT> &&
        common_size_simd_with<M, TF> &&
        (!basic_simd_mask_type<M> || !basic_simd_mask_type<TT> ||
            !basic_simd_mask_type<TF>) &&
        (!unqualified_select<common_abi_t<TT, TF, M>, M, TT, TF>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, TT tval, TF fval) noexcept
        -> common_size_simd_with<mask_result<M, TT, TF>> auto {
        return operator()(dx::to_basic_type(mask), dx::to_basic_type(tval),
            dx::to_basic_type(fval));
    }

    template <basic_simd_type TT, basic_simd_type TF, immediate_mask_for<TT> M>
    requires selectable_with<TT, TF> && same_abi_simd_as<TT, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr ternary_simd_t<TT, TF> DPL_VECTORCALL operator()(
        M mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF>;
        constexpr auto V = decltype(dx::to_immediate_mask<TT>(mask))::value;
        if constexpr (requires { select<V>(internal::abi<A>, tval, fval); }) {
            using R = decltype(select<V>(internal::abi<A>, tval, fval));
            static_assert(same_as<R, decltype(fallbacki<V>(tval, fval))>);
            if consteval {
                return fallbacki<V>(tval, fval);
            } else {
                return select<V>(internal::abi<A>, tval, fval);
            }
        } else {
            return fallbacki<V>(tval, fval);
        }
    }

    template <basic_simd_type TT, basic_simd_type TF, immediate_mask_for<TT> M>
    requires selectable_with<TT, TF> && only_unqualified<TT, TF> &&
        unqualified_selecti<common_abi_t<TT, TF>, M, TT, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, TT tval,
        TF fval) noexcept -> equivalent_simd_as<ternary_simd_t<TT, TF>> auto {
        using A = common_abi_t<TT, TF>;
        constexpr auto V = decltype(dx::to_immediate_mask<TT>(mask))::value;
        return select<V>(internal::abi<A>, tval, fval);
    }

    template <simd_type TT, simd_type TF, immediate_mask_for<TT> M>
    requires selectable_with<TT, TF> &&
        (!basic_simd_type<TT> || !basic_simd_type<TF>) &&
        (!unqualified_selecti<common_abi_t<TT, TF>, M, TT, TF>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(M mask, TT tval,
        TF fval) noexcept -> equivalent_simd_as<ternary_simd_t<TT, TF>> auto {
        using A = common_abi_t<TT, TF>;
        constexpr auto V = decltype(dx::to_immediate_mask<TT>(mask))::value;
        if constexpr (requires { select<V>(internal::abi<A>, tval, fval); }) {
            return select<V>(internal::abi<A>, tval, fval);
        } else {
            return operator()(
                mask, dx::to_basic_type(tval), dx::to_basic_type(fval));
        }
    }

    template <basic_simd_mask_type TT, basic_simd_mask_type TF,
        immediate_mask_for<TT> M>
    requires common_size_simd_with<TT, TF> && same_abi_simd_as<TT, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr common_size_type_t<TT, TF> DPL_VECTORCALL operator()(
        M mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF>;
        constexpr auto V = decltype(dx::to_immediate_mask<TT>(mask))::value;
        if constexpr (requires { select<V>(internal::abi<A>, tval, fval); }) {
            if consteval {
                return fallbacki<V>(tval, fval);
            } else {
                return select<V>(internal::abi<A>, tval, fval);
            }
        } else {
            return fallbacki<V>(tval, fval);
        }
    }

    template <simd_mask_type TT, simd_mask_type TF, immediate_mask_for<TT> M>
    requires common_size_simd_with<TT, TF> && only_unqualified<TT, TF> &&
        unqualified_selecti<common_abi_t<TT, TF>, M, TT, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, TT tval, TF fval) noexcept
        -> common_size_simd_with<common_size_type_t<TT, TF>> auto {
        using A = common_abi_t<TT, TF>;
        constexpr auto V = decltype(dx::to_immediate_mask<TT>(mask))::value;
        return select<V>(internal::abi<A>, tval, fval);
    }

    template <simd_mask_type TT, simd_mask_type TF, immediate_mask_for<TT> M>
    requires common_size_simd_with<TT, TF> &&
        (!basic_simd_mask_type<TT> || !basic_simd_mask_type<TF>) &&
        (!unqualified_selecti<common_abi_t<TT, TF>, M, TT, TF>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(
        M mask, TT tval, TF fval) noexcept
        -> common_size_simd_with<common_size_type_t<TT, TF>> auto {
        return operator()(
            mask, dx::to_basic_type(tval), dx::to_basic_type(fval));
    }

    template <basic_simd_mask_type M, basic_simd_class TT,
        broadcastable_to<TT> TF>
    requires same_abi_simd_as<M, TT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(M mask, TT tval, TF fval) noexcept
        -> equivalent_simd_as<TT> auto {
        if constexpr (requires {
                          select(internal::abi<M>, mask, tval, fval);
                      }) {
            if consteval {
                return operator()(mask, tval, dx::broadcast<TT>(fval));
            } else {
                return select(internal::abi<M>, mask, tval, fval);
            }
        } else {
            return operator()(mask, tval, dx::broadcast<TT>(fval));
        }
    }

    template <basic_simd_mask_type M, basic_simd_class TT,
        broadcastable_to<TT> TF>
    requires common_abi_simd_with<M, TT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(M mask, TT tval, TF fval) noexcept
        -> equivalent_simd_as<TT> auto {
        using A = common_abi_t<M, TT>;
        if constexpr (requires {
                          select(internal::abi<A>, mask, tval, fval);
                      }) {
            return select(internal::abi<A>, mask, tval, fval);
        } else {
            return operator()(mask, tval, dx::broadcast<TT>(fval));
        }
    }

    template <basic_simd_mask_type M, basic_simd_class TF,
        broadcastable_to<TF> TT>
    requires same_abi_simd_as<M, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(M mask, TT tval, TF fval) noexcept
        -> equivalent_simd_as<TF> auto {
        if constexpr (requires {
                          select(internal::abi<M>, mask, tval, fval);
                      }) {
            if consteval {
                return operator()(mask, dx::broadcast<TF>(tval), fval);
            } else {
                return select(internal::abi<M>, mask, tval, fval);
            }
        } else {
            return operator()(mask, dx::broadcast<TF>(tval), fval);
        }
    }

    template <basic_simd_mask_type M, basic_simd_class TF,
        broadcastable_to<TF> TT>
    requires common_abi_simd_with<M, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto operator()(M mask, TT tval, TF fval) noexcept
        -> equivalent_simd_as<TF> auto {
        using A = common_abi_t<M, TF>;
        if constexpr (requires {
                          select(internal::abi<A>, mask, tval, fval);
                      }) {
            return select(internal::abi<A>, mask, tval, fval);
        } else {
            return operator()(mask, dx::broadcast<TF>(tval), fval);
        }
    }

    template <basic_simd_mask_type M, broadcastable_to<to_simd_type_t<M>> TF,
        broadcastable_to<to_simd_type_t<M>> TT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, TT tval, TF fval) noexcept
        -> equivalent_simd_as<to_simd_type_t<M>> auto {
        using T = to_simd_type_t<M>;
        if constexpr (unqualified_select<M, M, TT, TF>) {
            if consteval {
                return operator()(
                    mask, dx::broadcast<T>(tval), dx::broadcast<T>(fval));
            } else {
                return select(internal::abi<M>, mask, tval, fval);
            }
        } else {
            return operator()(
                mask, dx::broadcast<T>(tval), dx::broadcast<T>(fval));
        }
    }

    template <simd_mask_type M, simd_class TT, broadcastable_to<TT> TF>
    requires common_abi_simd_with<M, TT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, TT tval, TF fval) noexcept
        -> equivalent_simd_as<TT> auto {
        using A = common_abi_t<M, TT>;
        if constexpr (unqualified_select<A, M, TT, TF>) {
            return select(internal::abi<A>, mask, tval, fval);
        } else {
            return operator()(
                dx::to_basic_type(mask), dx::to_basic_type(tval), fval);
        }
    }

    template <simd_mask_type M, simd_class TF, broadcastable_to<TF> TT>
    requires common_abi_simd_with<M, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, TT tval, TF fval) noexcept
        -> equivalent_simd_as<TF> auto {
        using A = common_abi_t<M, TF>;
        if constexpr (unqualified_select<A, M, TT, TF>) {
            return select(internal::abi<A>, mask, tval, fval);
        } else {
            return operator()(
                dx::to_basic_type(mask), tval, dx::to_basic_type(fval));
        }
    }

    template <simd_mask_type M, broadcastable_to<to_simd_type_t<M>> TF,
        broadcastable_to<to_simd_type_t<M>> TT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr auto operator()(M mask, TT tval, TF fval) noexcept
        -> equivalent_simd_as<to_simd_type_t<M>> auto {
        using T = to_simd_type_t<M>;
        if constexpr (unqualified_select<M, M, TT, TF>) {
            return select(internal::abi<M>, mask, tval, fval);
        } else {
            return operator()(dx::to_basic_type(mask), tval, fval);
        }
    }
};

template <auto V>
struct selecti_t {};
template <integral auto V>
struct selecti_t<V> : binary_operation_base<selecti_t<V>> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = immediate_mask<element_count<T>, V>;

public:
    template <simd_class T, simd_class F>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<select_t, mask_type<T>, T, F>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL operator()(T tval, F fval) noexcept {
        constexpr mask_type<T> mask{};
        return select_t::operator()(mask, tval, fval);
    }

    using binary_operation_base<selecti_t>::operator();
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <integral auto V>
inline constexpr internal::selecti_t<V> selecti{};
DPL_EXPORT inline constexpr internal::select_t select{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
