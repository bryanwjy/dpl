// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/concepts/immediate_like.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/concepts/semiregular.h"
#  include "dpl/std/functional/invoke.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/to_underlying.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace fpfix::internal {
enum class fpc : unsigned char {
    none = 0,
    zero = 1 << 0,
    one = 1 << 1,
    positive = 1 << 2,
    negative = 1 << 3,
    pos_inf = 1 << 4,
    neg_inf = 1 << 5,
    qnan = 1 << 6,
    snan = 1 << 7,
    all = (1 << 8) - 1,
    infinity = 0b11 << 4,
    nan = 0b11 << 6,
    finite = 0b1111,
};

consteval fpc operator|(fpc lhs, fpc rhs) noexcept {
    return static_cast<fpc>(
        __DPL to_underlying(lhs) | __DPL to_underlying(rhs));
}

consteval fpc operator&(fpc lhs, fpc rhs) noexcept {
    return static_cast<fpc>(
        __DPL to_underlying(lhs) & __DPL to_underlying(rhs));
}

consteval fpc operator~(fpc val) noexcept {
    return static_cast<fpc>(~__DPL to_underlying(val)) & fpc::all;
}

template <fpc C>
struct fpclass final {
    explicit consteval fpclass() noexcept = default;

    __DPL_HIDE_FROM_ABI constexpr ~fpclass() noexcept = default;

    template <fpc C2>
    consteval auto operator|(this fpclass, fpclass<C2>) noexcept {
        return fpclass<C | C2>{};
    }

    template <fpc C2>
    consteval auto operator&(this fpclass, fpclass<C2>) noexcept {
        return fpclass<C & C2>{};
    }

    template <fpc C2>
    consteval bool operator==(this fpclass, fpclass<C2>) noexcept {
        return C == C2;
    }

    consteval auto operator~(this fpclass) noexcept { return fpclass<~C>{}; }

    explicit consteval operator bool(this fpclass) noexcept {
        return C != fpc::none;
    }

    explicit consteval operator fpc(this fpclass) noexcept { return C; }
};

template <typename>
inline constexpr bool is_fpclass = false;
template <fpc C>
inline constexpr bool is_fpclass<fpclass<C>> = true;

template <typename...>
class set;

template <floating_point E, fpc C>
struct fix_case {
public:
    explicit consteval fix_case() noexcept = default;
};

template <typename T>
inline constexpr bool is_dependent_result = false;

struct signed_inf_t final {
    explicit consteval signed_inf_t() noexcept = default;
    consteval bool operator==(signed_inf_t) noexcept { return true; }
    consteval bool operator==(auto) noexcept { return false; }
};

struct revert_t final {
    explicit consteval revert_t() noexcept = default;
    consteval bool operator==(revert_t) noexcept { return true; }
    consteval bool operator==(auto) noexcept { return false; }
};

template <>
inline constexpr bool is_dependent_result<signed_inf_t> = false;
template <>
inline constexpr bool is_dependent_result<revert_t> = false;

template <typename T>
concept dependent_result = is_dependent_result<T>;
template <typename T, typename E>
concept result_type = floating_point<E> &&
    (__DPL datapar::internal::immediate_like_of<T, E> || dependent_result<T>);

template <fpc C, typename T>
class pair final {
public:
    explicit consteval pair() noexcept = default;

    template <fpc CR, typename TR>
    requires ((CR & C) == fpc::none)
    consteval auto operator|(this pair, pair<CR, TR>) noexcept {
        if constexpr (same_as<T, TR>) {
            return set<pair<C | CR, T>>{};
        } else {
            return set<pair, pair<CR, TR>>{};
        }
    }

    template <floating_point E, fpc CT>
    requires result_type<T, E>
    static consteval auto operator[](fix_case<E, CT>) noexcept {
        static_assert((C & CT) != fpc::none);
        if constexpr (same_as<T, revert_t>) {
            return revert_t{};
        } else if constexpr (same_as<T, signed_inf_t>) {
            return signed_inf_t{};
        } else {
            return datapar::to_immediate<E>(T{});
        }
    }

    template <fpc CT>
    static consteval auto operator[](fpclass<CT>) noexcept {
        static_assert((C & CT) != fpc::none);
        if constexpr (same_as<T, revert_t>) {
            return revert_t{};
        } else if constexpr (same_as<T, signed_inf_t>) {
            return signed_inf_t{};
        } else {
            return T{};
        }
    }
};

template <fpc... Cs, typename... Ts>
class set<pair<Cs, Ts>...> final {
    static constexpr fpc all = (fpc::none | ... | Cs);

public:
    explicit consteval set() noexcept = default;

    template <fpc CR, typename TR>
    requires ((all & CR) == fpc::none)
    consteval auto operator|(this set self, pair<CR, TR> rhs) noexcept {
        return [rhs]<fpc C0, typename T0, fpc... Cn, typename... Tn>(
                   this auto self, pair<C0, T0> head, pair<Cn, Tn>... tail) {
            if constexpr (same_as<T0, TR>) {
                return pair<C0 | CR, T0>{} | set<pair<Cn, Ts>...>{};
            } else if constexpr (sizeof...(Cn) == 0) {
                return rhs;
            } else {
                return head | self(tail...);
            }
        }(pair<Cs, Ts>()...);
    }

    template <fpc CL, typename TL>
    requires ((all & CL) == fpc::none)
    friend consteval auto operator|(pair<CL, TL> lhs, set self) noexcept {
        return []<fpc C0, typename T0, fpc... Cm, typename... Tm, fpc... Cn,
                   typename... Tn>(this auto self,
                   set<pair<C0, T0>, pair<Cm, Tm>...>, set<pair<Cn, Tn>...>) {
            if constexpr (same_as<T0, TL>) {
                return set<pair<Cn, Tn>..., pair<C0 | CL, T0>,
                    pair<Cm, Tm>...>{};
            } else if constexpr (sizeof...(Cm) == 0) {
                return set<pair<Cn, Tn>..., pair<C0, T0>>{};
            } else {
                return self(set<pair<Cm, Tm>...>{},
                    set<pair<Cn, Tn>..., pair<C0, T0>>{});
            }
        }(self, set<>{});
    }

    template <fpc... CR, typename... TR>
    requires ((all & (fpc::none | ... | CR)) == fpc::none)
    consteval auto operator|(this set self, set<pair<Cs, Ts>...>) noexcept {
        return (self | ... | pair<Cs, Ts>{});
    }

    template <floating_point E, fpc CT>
    requires (... && result_type<Ts, E>)
    static consteval auto operator[](fix_case<E, CT> arg) noexcept {
        static_assert((all & CT) == CT);
        return [arg]<fpc C0, typename T0, fpc... Cn, typename... Tn>(
                   this auto self, pair<C0, T0> head, pair<Cn, Tn>... tail) {
            if constexpr ((C0 & CT) != fpc::none) {
                return head[arg];
            } else if constexpr (sizeof...(Cs) > 0) {
                return self(tail...);
            } else {
                static_assert(sizeof...(Cs) > 0);
                return nullptr;
            }
        }(pair<Cs, Ts>()...);
    }

    template <fpc CT>
    static consteval auto operator[](fpclass<CT> arg) noexcept {
        static_assert((all & CT) == CT);
        return [arg]<fpc C0, typename T0, fpc... Cn, typename... Tn>(
                   this auto self, pair<C0, T0> head, pair<Cn, Tn>... tail) {
            if constexpr ((C0 & CT) != fpc::none) {
                return head[arg];
            } else if constexpr (sizeof...(Cs) > 0) {
                return self(tail...);
            } else {
                static_assert(sizeof...(Cs) > 0);
                return nullptr;
            }
        }(pair<Cs, Ts>()...);
    }
};

struct template_for_t final {

    template <fpc C, semiregular T, regular_invocable<fpclass<C>, T> F>
    static consteval void operator()(F&& enumerator, pair<C, T>) noexcept {
        constexpr fpclass<C> condition{};
        constexpr T value{};
        __DPL invoke(__DPL forward<F>(enumerator), condition, value);
    }

    template <fpc... Cs, semiregular... Ts, typename F>
    requires (... && regular_invocable<F, fpclass<Cs>, Ts>)
    static consteval void operator()(
        F&& enumerator, set<pair<Cs, Ts>...> arg) noexcept {
        [&enumerator]<fpc C0, typename T0, fpc... Cn, typename... Tn>(
            this auto self, pair<C0, T0> head, pair<Cn, Tn>... tail) {
            template_for_t::operator()(enumerator, head);
            if constexpr (sizeof...(Cn) > 0) {
                self(tail...);
            }
        }(pair<Cs, Ts>{}...);
    }
};

template <typename>
inline constexpr bool is_condition_set = false;
template <fpc C, typename T>
inline constexpr bool is_condition_set<pair<C, T>> = true;
template <fpc... Cs, typename... Ts>
inline constexpr bool is_condition_set<set<pair<Cs, Ts>...>> = true;

template <typename... Vs, fpc C, typename T>
consteval bool subset_of(pair<C, T>) noexcept {
    return (... || same_as<T, Vs>);
}

template <typename... Vs, fpc... Cs, typename... Ts>
consteval bool subset_of(set<pair<Cs, Ts>...>) noexcept {
    return (... && internal::subset_of<Vs...>(pair<Cs, Ts>{}));
}

template <typename, typename>
inline constexpr bool is_condition_set_for = false;
template <typename E, fpc C, typename T>
inline constexpr bool is_condition_set_for<pair<C, T>, E> = result_type<T, E>;
template <typename E, fpc... Cs, typename... Ts>
inline constexpr bool is_condition_set_for<set<pair<Cs, Ts>...>, E> =
    (... && result_type<Ts, E>);
} // namespace fpfix::internal

namespace fpfix {

DPL_EXPORT inline constexpr internal::revert_t revert{};
DPL_EXPORT inline constexpr internal::signed_inf_t signed_inf{};

DPL_EXPORT template <typename T>
concept fpclass_type = internal::is_fpclass<T>;
DPL_EXPORT template <typename T>
concept condition_set = internal::is_condition_set<T>;
DPL_EXPORT template <typename T, typename E>
concept condition_set_for = floating_point<E> && condition_set<T> &&
    internal::is_condition_set_for<T, E>;

DPL_EXPORT template <typename T, typename E, auto... Rs>
concept result_subset_of = condition_set_for<T, E> &&
    (... && internal::result_type<decltype(Rs), E>) &&
    internal::subset_of<decltype(Rs)...>(T{});

DPL_EXPORT inline constexpr internal::template_for_t template_for{};

// clang-format off
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::none> none{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::zero> zero{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::one> one{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::positive> positive{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::negative> negative{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::pos_inf> pos_inf{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::neg_inf> neg_inf{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::qnan> qnan{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::snan> snan{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::all> all{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::infinity> infinity{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::nan> nan{};
DPL_EXPORT inline constexpr internal::fpclass<internal::fpc::finite> finite{};
// clang-format on
template <internal::fpc C>
consteval int popcount(internal::fpclass<C>) noexcept {
    return __DPL popcount(__DPL to_underlying(C));
}

DPL_EXPORT template <fpclass_type auto C, auto V>
inline constexpr internal::pair<static_cast<internal::fpc>(C), decltype(V)>
    condition{};

} // namespace fpfix
} // namespace datapar

DPL_EXPORT namespace fpfix = datapar::fpfix; // NOLINT

DPL_DEFAULT_NAMESPACE_END