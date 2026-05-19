// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/operation_base.h"
#include "dpl/core/operations/transform.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcast.h"
#  include "dpl/core/basic/const_mask.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/compatible_mask_with.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/core/type_traits/make_simd_mask_type.h"
#  include "dpl/core/type_traits/make_simd_type.h"
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
    __DPL decay_t<decltype(false ? __DPL declval<L>()
                                   : __DPL declval<R>())>;

template <typename L, typename R>
struct ternary_type {};
template <typename L, typename R>
using ternary_type_t DPL_NODEBUG = typename ternary_type<L, R>::type;

template <simd_vector L, simd_element R>
struct ternary_type<L, R> : ternary_type<simd_lane_representation_t<L>, R> {};

template <simd_element L, simd_vector R>
struct ternary_type<L, R> : ternary_type<L, simd_lane_representation_t<R>> {};

template <simd_vector L, simd_vector R>
struct ternary_type<L, R> :
    ternary_type<simd_lane_representation_t<L>, simd_lane_representation_t<R>> {
};

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

template <simd_vector L, simd_vector R>
requires requires { typename ternary_type_t<L, R>; }
struct ternary_simd<L, R> {
    using type DPL_NODEBUG =
        basic_vector<ternary_type_t<L, R>, common_abi_t<L, R>>;
};

template <typename R, typename L, typename C = make_simd_mask_type_t<L>>
concept selectable_with = common_size_simd_with<R, L> &&
    compatible_mask_with<C, R> && compatible_mask_with<C, L> && requires {
        typename ternary_simd_t<L, R>;
        requires common_size_with<ternary_simd_t<L, R>,
            common_size_type_t<R, L>>;
    } && same_abi_as<common_abi_t<L, R>, typename C::abi_type>;

template <typename R, typename L>
concept selectablei_with = common_size_simd_with<R, L> && requires {
    typename ternary_simd_t<L, R>;
    requires common_size_with<ternary_type_t<L, R>, common_size_type_t<R, L>>;
};

template <typename R, typename L, typename C = make_simd_mask_type_t<L>>
concept mselectable_with = common_size_simd_with<R, L> &&
    common_size_simd_with<C, R> && common_size_simd_with<C, L> &&
    same_abi_simd_as<common_size_simd_t<L, R>, C>;

template <typename T, typename L, typename R, typename A>
concept ternary_result_of = simd_with<T,
    ternary_type_t<typename L::value_type, typename R::value_type>, A>;

template <typename T, typename L, typename R, typename A>
concept mternary_result_of = simd_with<T,
    common_size_type_t<typename L::value_type, typename R::value_type>, A>;

template <typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_select = requires(C cond, L lhs, R rhs) {
    { select(internal::abi<A>, cond, lhs, rhs) } -> ternary_result_of<L, R, A>;
};

template <typename C, typename L, typename R,
    typename A = common_abi_t<L, R, C>>
concept unqualified_mselect = requires(C cond, L lhs, R rhs) {
    { select(internal::abi<A>, cond, lhs, rhs) } -> mternary_result_of<L, R, A>;
};

template <typename C, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_selecti = const_mask_for<C, L> && requires(L lhs, R rhs) {
    {
        select<const_mask_v<L, C>>(internal::abi<A>, lhs, rhs)
    } -> ternary_result_of<L, R, A>;
};

template <typename C, typename L, typename R, typename A = common_abi_t<L, R>>
concept unqualified_mselecti = const_mask_for<C, L> && requires(L lhs, R rhs) {
    {
        select<const_mask_v<L, C>>(internal::abi<A>, lhs, rhs)
    } -> mternary_result_of<L, R, A>;
};

struct select_t {
private:
    template <typename M, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_mask<M, A> mask,
        basic_vector<ET, A> tval, basic_vector<EF, A> fval) noexcept {
        using ER = ternary_type_t<ET, EF>;
        return internal::transform<basic_vector<ER, A>>(
            [](bool cond, ET tval, EF fval) -> ER {
                return cond ? tval : fval;
            },
            mask, tval, fval);
    }

    template <typename M, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallback(basic_mask<M, A> mask,
        basic_mask<ET, A> tval, basic_mask<EF, A> fval) noexcept {
        using ER = common_size_type_t<M, ET, EF>;
        return internal::transform<basic_mask<ER, A>>(
            [](bool cond, bool tval, bool fval) -> bool {
                return cond ? tval : fval;
            },
            mask, tval, fval);
    }

    template <auto V, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_vector<ET, A> tval, basic_vector<EF, A> fval) noexcept {
        static constexpr const_mask<simd_abi_traits<ET, A>::size, V> mask{};
        using ER = ternary_type_t<ET, EF>;
        return internal::itransform<basic_vector<ER, A>>(
            [](auto idx, ET tval, EF fval) -> ER {
                return mask[idx] ? tval : fval;
            },
            tval, fval);
    }

    template <auto V, typename ET, typename EF, typename A>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr auto DPL_VECTORCALL fallbacki(
        basic_mask<ET, A> tval, basic_mask<EF, A> fval) noexcept {
        static constexpr const_mask<simd_abi_traits<ET, A>::size, V> mask{};
        using ER = common_size_type_t<ET, EF>;
        return internal::itransform<basic_mask<ER, A>>(
            [](auto idx, ET tval, EF fval) -> ER {
                return mask[idx] ? tval : fval;
            },
            tval, fval);
    }

public:
    template <simd_vector TT, compatible_mask_with<TT> C,
        selectable_with<TT, C> TF>
    requires same_abi_simd_as<TT, TF> && fixed_width_abi<common_abi_t<TT, TF>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = typename C::abi_type;
        if constexpr (unqualified_select<C, TT, TF, A>) {
            if constexpr (canonical_vector<TT> && canonical_vector<TF> &&
                canonical_mask<C>) {
                if consteval {
                    return fallback(mask, tval, fval);
                } else {
                    return select(internal::abi<A>, mask, tval, fval);
                }
            } else {
                return select(internal::abi<A>, mask, tval, fval);
            }
        } else if constexpr (canonical_vector<TT> && canonical_vector<TF> &&
            canonical_mask<C>) {
            return fallback(mask, tval, fval);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(tval),
                dx::to_canonical(fval));
        }
    }

    template <simd_vector TT, compatible_mask_with<TT> C,
        selectable_with<TT, C> TF>
    requires (!same_abi_simd_as<TT, TF> ||
                 scalable_abi<common_abi_t<TT, TF>>) &&
        (unqualified_select<C, TT, TF> ||
            unqualified_select<C, canonical_type_t<TT>, canonical_type_t<TF>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF>;
        if constexpr (unqualified_select<C, TT, TF>) {
            return select(internal::abi<A>, mask, tval, fval);
        } else {
            return select(internal::abi<A>, dx::to_canonical(mask),
                dx::to_canonical(tval), dx::to_canonical(fval));
        }
    }

    template <simd_mask TT, common_size_simd_with<TT> C,
        mselectable_with<TT, C> TF>
    requires same_abi_simd_as<TT, TF> && fixed_width_abi<common_abi_t<TT, TF>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = typename C::abi_type;
        if constexpr (unqualified_mselect<C, TT, TF, A>) {
            if constexpr (canonical_mask<TT> && canonical_mask<TF> &&
                canonical_mask<C>) {
                if consteval {
                    return fallback(mask, tval, fval);
                } else {
                    return select(internal::abi<A>, mask, tval, fval);
                }
            } else {
                return select(internal::abi<A>, mask, tval, fval);
            }
        } else if constexpr (canonical_mask<TT> && canonical_mask<TF> &&
            canonical_mask<C>) {
            return fallback(mask, tval, fval);
        } else {
            return operator()(dx::to_canonical(mask), dx::to_canonical(tval),
                dx::to_canonical(fval));
        }
    }

    template <simd_mask TT, common_size_simd_with<TT> C,
        mselectable_with<TT, C> TF>
    requires (!same_abi_simd_as<TT, TF> ||
                 scalable_abi<common_abi_t<TT, TF>>) &&
        (unqualified_mselect<C, TT, TF> ||
            unqualified_mselect<C, canonical_type_t<TT>, canonical_type_t<TF>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF>;
        if constexpr (unqualified_mselect<C, TT, TF>) {
            return select(internal::abi<A>, mask, tval, fval);
        } else {
            return select(internal::abi<A>, dx::to_canonical(mask),
                dx::to_canonical(tval), dx::to_canonical(fval));
        }
    }

    template <simd_vector TT, selectablei_with<TT> TF, const_mask_for<TT> C>
    requires same_abi_simd_as<TT, TF> && fixed_width_abi<common_abi_t<TT, TF>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = typename TT::abi_type;
        constexpr auto V =
            decltype(dx::to_compatible_const_mask<TT>(mask))::value;
        if constexpr (unqualified_selecti<C, TT, TF, A>) {
            if constexpr (canonical_vector<TT> && canonical_vector<TF>) {
                if consteval {
                    return fallbacki<V>(tval, fval);
                } else {
                    return select<V>(internal::abi<A>, tval, fval);
                }
            } else {
                return select<V>(internal::abi<A>, tval, fval);
            }
        } else if constexpr (canonical_vector<TT> && canonical_vector<TF>) {
            return fallbacki<V>(tval, fval);
        } else {
            return operator()(
                mask, dx::to_canonical(tval), dx::to_canonical(fval));
        }
    }

    template <simd_vector TT, selectablei_with<TT> TF, const_mask_for<TT> C>
    requires (!same_abi_simd_as<TT, TF> ||
                 scalable_abi<common_abi_t<TT, TF>>) &&
        (unqualified_selecti<C, TT, TF> ||
            unqualified_selecti<C, canonical_type_t<TT>, canonical_type_t<TF>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF>;
        constexpr auto V =
            decltype(dx::to_compatible_const_mask<TT>(mask))::value;
        if constexpr (unqualified_selecti<C, TT, TF>) {
            return select<V>(internal::abi<A>, tval, fval);
        } else {
            return select<V>(internal::abi<A>, dx::to_canonical(tval),
                dx::to_canonical(fval));
        }
    }

    template <simd_mask TT, common_size_simd_with<TT> TF, const_mask_for<TT> C>
    requires same_abi_simd_as<TT, TF>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF>;
        constexpr auto V =
            decltype(dx::to_compatible_const_mask<TT>(mask))::value;
        if constexpr (unqualified_mselecti<C, TT, TF>) {
            if constexpr (canonical_mask<TT> && canonical_mask<TF>) {
                if consteval {
                    return fallbacki<V>(tval, fval);
                } else {
                    return select<V>(internal::abi<A>, tval, fval);
                }
            } else {
                return select<V>(internal::abi<A>, tval, fval);
            }
        } else if constexpr (canonical_mask<TT> && canonical_mask<TF>) {
            return fallbacki<V>(tval, fval);
        } else {
            return operator()(
                mask, dx::to_canonical(tval), dx::to_canonical(fval));
        }
    }

    template <simd_mask TT, common_size_simd_with<TT> TF, const_mask_for<TT> C>
    requires (!same_abi_simd_as<TT, TF>) &&
        (unqualified_mselecti<C, TT, TF> ||
            unqualified_mselecti<C, canonical_type_t<TT>, canonical_type_t<TF>>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = common_abi_t<TT, TF>;
        constexpr auto V =
            decltype(dx::to_compatible_const_mask<TT>(mask))::value;
        if constexpr (unqualified_mselecti<C, TT, TF>) {
            return select<V>(internal::abi<A>, tval, fval);
        } else {
            return select<V>(internal::abi<A>, dx::to_canonical(tval),
                dx::to_canonical(fval));
        }
    }

    template <simd_mask C, simd_class TT, broadcastable_to<TT> TF>
    requires selectable_with<TT, TT, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = typename C::abi_type;
        if constexpr (requires {
                          {
                              select(internal::abi<A>, mask, tval, fval)
                          } -> equivalent_simd_as<TT>;
                      }) {
            if constexpr (canonical_mask<C> && canonical_class<TT>) {
                if consteval {
                    return operator()(mask, tval, dx::broadcast<TT>(fval));
                } else {
                    return select(internal::abi<A>, mask, tval, fval);
                }
            } else {
                return select(internal::abi<A>, mask, tval, fval);
            }
        } else {
            return operator()(mask, tval, dx::broadcast<TT>(fval));
        }
    }

    template <simd_mask C, simd_class TF, broadcastable_to<TF> TT>
    requires selectable_with<TF, TF, C>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using A = typename C::abi_type;
        if constexpr (requires {
                          {
                              select(internal::abi<A>, mask, tval, fval)
                          } -> equivalent_simd_as<TT>;
                      }) {
            if constexpr (canonical_mask<C> && canonical_class<TT>) {
                if consteval {
                    return operator()(mask, dx::broadcast<TF>(tval), fval);
                } else {
                    return select(internal::abi<A>, mask, tval, fval);
                }
            } else {
                return select(internal::abi<A>, mask, tval, fval);
            }
        } else {
            return operator()(mask, dx::broadcast<TF>(tval), fval);
        }
    }

    template <simd_mask C, broadcastable_to<make_simd_type_t<C>> TF,
        broadcastable_to<make_simd_type_t<C>> TT>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(C mask, TT tval, TF fval) noexcept {
        using T = make_simd_type_t<C>;
        using A = typename C::abi_type;
        if constexpr (unqualified_select<C, TT, TF, A>) {
            if constexpr (canonical_mask<C>) {
                if consteval {
                    return operator()(
                        mask, dx::broadcast<T>(tval), dx::broadcast<T>(fval));
                } else {
                    return select(internal::abi<A>, mask, tval, fval);
                }
            } else {
                return select(internal::abi<A>, mask, tval, fval);
            }
        } else {
            return operator()(
                mask, dx::broadcast<T>(tval), dx::broadcast<T>(fval));
        }
    }
};

template <auto V>
struct selecti_t {};
template <integral auto V>
struct selecti_t<V> : binary_operation_base<selecti_t<V>> {
private:
    template <typename T>
    using mask_type DPL_NODEBUG = make_const_mask_t<T, V>;

public:
    template <simd_class T, simd_class F>
    requires requires {
        typename mask_type<T>;
        requires regular_invocable<select_t, mask_type<T>, T, F>;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T tval, F fval) noexcept {
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
