// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/details/fwd.h" // IWYU pragma: keep

#if !DPL_MODULES

#  include "dpl/core/concepts/cpo_invocable.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/numbers/details/binary_layout_floating_point.h"
#  include "dpl/std/concepts/semiregular.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/to_underlying.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
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
class fixup_set;

namespace dxn = __DPL details::numbers;

template <dxn::binary_layout_floating_point E, fpc C>
struct fix_case {
public:
    explicit consteval fix_case() noexcept = default;
};

template <typename T>
inline constexpr bool is_dependent_result = false;

struct signed_inf_t {
    explicit consteval signed_inf_t() noexcept = default;
    consteval bool operator==(signed_inf_t) noexcept { return true; }
    consteval bool operator==(auto) noexcept { return false; }
};

struct revert_t {
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
concept result_type = dxn::binary_layout_floating_point<E> &&
    (__DPL datapar::internal::immediate_like_of<T, E> || dependent_result<T>);

template <fpc C, typename T>
class fixup_pair {
public:
    explicit consteval fixup_pair() noexcept = default;

    template <fpc CR, typename TR>
    requires ((CR & C) == fpc::none)
    consteval auto operator|(this fixup_pair, fixup_pair<CR, TR>) noexcept {
        if constexpr (same_as<T, TR>) {
            return fixup_set<fixup_pair<C | CR, T>>{};
        } else {
            return fixup_set<fixup_pair, fixup_pair<CR, TR>>{};
        }
    }

    template <dxn::binary_layout_floating_point E, fpc CT>
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
class fixup_set<fixup_pair<Cs, Ts>...> {
    static constexpr fpc all = (fpc::none | ... | Cs);

public:
    explicit consteval fixup_set() noexcept = default;

    template <fpc CR, typename TR>
    requires ((all & CR) == fpc::none)
    consteval auto operator|(
        this fixup_set self, fixup_pair<CR, TR> rhs) noexcept {
        return [rhs]<fpc... Cn, typename... Tn, fpc C0, typename T0, fpc... Cm,
                   typename... Tm>(this auto self,
                   fixup_set<fixup_pair<Cn, Tn>...>, fixup_pair<C0, T0>,
                   fixup_pair<Cm, Tm>... tail) {
            if constexpr (same_as<T0, TR>) {
                return fixup_set<fixup_pair<Cn, Tn>..., fixup_pair<C0 | CR, T0>,
                    fixup_pair<Cm, Tm>...>{};
            } else if constexpr (sizeof...(Cm) == 0) {
                return fixup_set<fixup_pair<Cn, Tn>..., fixup_pair<C0, T0>,
                    fixup_pair<CR, TR>>{};
            } else {
                return self(
                    fixup_set<fixup_pair<Cn, Tn>..., fixup_pair<C0, T0>>{},
                    tail...);
            }
        }(fixup_set<>{}, fixup_pair<Cs, Ts>()...);
    }

    template <fpc CL, typename TL>
    requires ((all & CL) == fpc::none)
    friend consteval auto operator|(
        fixup_pair<CL, TL> lhs, fixup_set self) noexcept {
        return []<fpc C0, typename T0, fpc... Cm, typename... Tm, fpc... Cn,
                   typename... Tn>(this auto self,
                   fixup_set<fixup_pair<Cn, Tn>...>, fixup_pair<C0, T0>,
                   fixup_pair<Cm, Tm>... next) {
            if constexpr (same_as<T0, TL>) {
                return fixup_set<fixup_pair<Cn, Tn>..., fixup_pair<C0 | CL, T0>,
                    fixup_pair<Cm, Tm>...>{};
            } else if constexpr (sizeof...(Cm) == 0) {
                return fixup_set<fixup_pair<CL, TL>, fixup_pair<Cn, Tn>...,
                    fixup_pair<C0, T0>>{};
            } else {
                return self(
                    fixup_set<fixup_pair<Cn, Tn>..., fixup_pair<C0, T0>>{},
                    next...);
            }
        }(fixup_set<>{}, fixup_pair<Cs, Ts>()...);
    }

    template <fpc... CR, typename... TR>
    requires ((all & (fpc::none | ... | CR)) == fpc::none)
    consteval auto operator|(
        this fixup_set self, fixup_set<fixup_pair<CR, TR>...>) noexcept {
        return (self | ... | fixup_pair<CR, TR>{});
    }

    template <dxn::binary_layout_floating_point E, fpc CT>
    requires (... && result_type<Ts, E>)
    static consteval auto operator[](fix_case<E, CT> arg) noexcept {
        static_assert((all & CT) == CT);
        return [arg]<fpc C0, typename T0, fpc... Cn, typename... Tn>(
                   this auto self, fixup_pair<C0, T0> head,
                   fixup_pair<Cn, Tn>... tail) {
            if constexpr ((C0 & CT) != fpc::none) {
                return head[arg];
            } else if constexpr (sizeof...(Cs) > 0) {
                return self(tail...);
            } else {
                static_assert(sizeof...(Cs) > 0);
                return nullptr;
            }
        }(fixup_pair<Cs, Ts>()...);
    }

    template <fpc CT>
    static consteval auto operator[](fpclass<CT> arg) noexcept {
        static_assert((all & CT) == CT);
        return [arg]<fpc C0, typename T0, fpc... Cn, typename... Tn>(
                   this auto self, fixup_pair<C0, T0> head,
                   fixup_pair<Cn, Tn>... tail) {
            if constexpr ((C0 & CT) != fpc::none) {
                return head[arg];
            } else if constexpr (sizeof...(Cs) > 0) {
                return self(tail...);
            } else {
                static_assert(sizeof...(Cs) > 0);
                return nullptr;
            }
        }(fixup_pair<Cs, Ts>()...);
    }
};

struct template_for_t final {

    template <fpc C, semiregular T,
        dx::internal::cpo_invocable<fpclass<C>, T> F>
    static consteval void operator()(
        F&& enumerator, fixup_pair<C, T>) noexcept {
        constexpr fpclass<C> condition{};
        constexpr T value{};
        __DPL forward<F>(enumerator)(condition, value);
    }

    template <fpc... Cs, semiregular... Ts, typename F>
    requires (... && dx::internal::cpo_invocable<F, fpclass<Cs>, Ts>)
    static consteval void operator()(
        F&& enumerator, fixup_set<fixup_pair<Cs, Ts>...> arg) noexcept {
        [&enumerator]<fpc C0, typename T0, fpc... Cn, typename... Tn>(
            this auto self, fixup_pair<C0, T0> head,
            fixup_pair<Cn, Tn>... tail) {
            template_for_t::operator()(enumerator, head);
            if constexpr (sizeof...(Cn) > 0) {
                self(tail...);
            }
        }(fixup_pair<Cs, Ts>{}...);
    }
};

template <typename>
inline constexpr bool is_condition_set = false;
template <fpc C, typename T>
inline constexpr bool is_condition_set<fixup_pair<C, T>> = true;
template <fpc... Cs, typename... Ts>
inline constexpr bool is_condition_set<fixup_set<fixup_pair<Cs, Ts>...>> = true;

template <typename... Vs, fpc C, typename T>
consteval bool subset_of(fixup_pair<C, T>) noexcept {
    return (... || same_as<T, Vs>);
}

template <typename... Vs, fpc... Cs, typename... Ts>
consteval bool subset_of(fixup_set<fixup_pair<Cs, Ts>...>) noexcept {
    return (... && fmath::subset_of<Vs...>(fixup_pair<Cs, Ts>{}));
}

template <typename, typename>
inline constexpr bool is_condition_set_for = false;
template <typename E, fpc C, typename T>
inline constexpr bool is_condition_set_for<fixup_pair<C, T>, E> =
    result_type<T, E>;
template <typename E, fpc... Cs, typename... Ts>
inline constexpr bool
    is_condition_set_for<fixup_set<fixup_pair<Cs, Ts>...>, E> =
        (... && result_type<Ts, E>);
} // namespace datapar::fmath

__DPL_DEFAULT_NAMESPACE_END
