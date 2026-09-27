// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

export module mpfr:jump_table;
import dpl;

export namespace mpfr {

#define _JT_FWD(...) static_cast<decltype(__VA_ARGS__)&&>(__VA_ARGS__)

template <typename T, T... Ns>
class jump_table;

template <typename T>
class jump_table<T> {
public:
    template <T N, T... Others>
    static constexpr jump_table<T, N, Others...> add_case() {
        return {};
    }
};
struct any_result {
    template <typename T>
    [[noreturn]] operator T&&() const noexcept {
        dpl::unreachable();
    }
    template <typename T>
    [[noreturn]] operator T&() const noexcept {
        dpl::unreachable();
    }
};

template <typename T>
concept voidable = dpl::is_void_v<T> || dpl::same_as<T, any_result>;

template <voidable... Ts>
requires (!(... && dpl::same_as<Ts, any_result>))
void make_multi_return() noexcept;

template <typename... Ts>
requires (!(... && voidable<Ts>) && !(... && dpl::same_as<Ts, any_result>))
auto make_multi_return() noexcept -> dpl::common_reference_t<Ts...>;

template <typename... Ts>
using multi_return_t DPL_NODEBUG = decltype(make_multi_return<Ts...>());

template <typename F, typename... Ts>
struct jump_result {
    using type DPL_NODEBUG = any_result;
};

template <typename F, typename... Ts>
requires dpl::invocable<F, Ts...>
struct jump_result<F, Ts...> {
    using type DPL_NODEBUG = dpl::invoke_result_t<F, Ts...>;
};

template <typename F, typename... Ts>
using jump_result_t = typename jump_result<F, Ts...>::type;

template <typename T, T... Ns>
class jump_table {
    using size_t = decltype(sizeof(0));
    static_assert(
        dpl::semiregular<T> && requires(T value) {
            typename dpl::integral_constant<T, (..., Ns)>;
            [](T value, dpl::integral_constant<T, (..., Ns)> first) {
                switch (value) {
                case (first.value):
                    return 0;
                default:
                    return 1;
                }
            }(value, {});
        }, "Invalid jump table index type");

    static constexpr size_t size = sizeof...(Ns);

    static consteval T index_to_value(size_t idx) noexcept {
        T const values[] = {Ns...};
        return values[idx];
    }

    template <size_t I>
    using ith_type DPL_NODEBUG = dpl::integral_constant<T, index_to_value(I)>;

public:
    template <typename F, typename... Args>
    using result_type DPL_NODEBUG = multi_return_t<jump_result_t<F, T, Args...>,
        jump_result_t<F, dpl::integral_constant<T, Ns>, Args...>...>;

private:
    template <typename F, typename... Args>
    static constexpr result_type<F, Args...> default_(
        F&& callable, T value, Args&&... args) {
        if constexpr (dpl::is_invocable_v<F, T, Args...>) {
            if (dpl::is_void_v<result_type<F, Args...>>) {
                dpl::invoke(_JT_FWD(callable), value, _JT_FWD(args)...);
            } else {
                return dpl::invoke(_JT_FWD(callable), value, _JT_FWD(args)...);
            }
        } else if constexpr (!dpl::is_void_v<result_type<F, Args...>>) {
            DPL_BUILTIN_unreachable();
        }
        // If the callable returns nothing, the default case can be ignored
    }

    template <size_t I, typename F, typename... Args>
    static constexpr auto case_(F&& callable, T value, Args&&... args)
        -> decltype(auto) {
        if constexpr (I < size) {
            if constexpr (dpl::is_invocable_v<F, ith_type<I>, Args...>) {
                constexpr ith_type<I> case_arg{};
                return dpl::invoke(
                    _JT_FWD(callable), case_arg, _JT_FWD(args)...);
            } else {
                return default_(_JT_FWD(callable), value, _JT_FWD(args)...);
            }
        } else {
            return default_(_JT_FWD(callable), value, _JT_FWD(args)...);
        }
    }

    template <size_t I, typename F, typename... Args>
    static constexpr auto next_(F&& callable, T value, Args&&... args)
        -> decltype(auto) {
        if constexpr (I < size) {
            return impl<I>(_JT_FWD(callable), value, _JT_FWD(args)...);
        } else {
            return default_(_JT_FWD(callable), value, _JT_FWD(args)...);
        }
    }

    template <size_t I, typename F, typename... Args>
    static constexpr auto impl(F&& callable, T value, Args&&... args)
        -> decltype(auto) {
#define __DPL_JT_CASE(X)        \
    case index_to_value(I + X): \
        return case_<I + X>(_JT_FWD(callable), value, _JT_FWD(args)...)
#define __DPL_JT_DEFAULT(X) \
    default:                \
        return next_<I + X>(_JT_FWD(callable), value, _JT_FWD(args)...)
        if constexpr (size >= 16 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_CASE(1);
                __DPL_JT_CASE(2);
                __DPL_JT_CASE(3);
                __DPL_JT_CASE(4);
                __DPL_JT_CASE(5);
                __DPL_JT_CASE(6);
                __DPL_JT_CASE(7);
                __DPL_JT_CASE(8);
                __DPL_JT_CASE(9);
                __DPL_JT_CASE(10);
                __DPL_JT_CASE(11);
                __DPL_JT_CASE(12);
                __DPL_JT_CASE(13);
                __DPL_JT_CASE(14);
                __DPL_JT_CASE(15);
                __DPL_JT_DEFAULT(16);
            }
        } else if constexpr (size >= 8 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_CASE(1);
                __DPL_JT_CASE(2);
                __DPL_JT_CASE(3);
                __DPL_JT_CASE(4);
                __DPL_JT_CASE(5);
                __DPL_JT_CASE(6);
                __DPL_JT_CASE(7);
                __DPL_JT_DEFAULT(8);
            }
        } else if constexpr (size == 7 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_CASE(1);
                __DPL_JT_CASE(2);
                __DPL_JT_CASE(3);
                __DPL_JT_CASE(4);
                __DPL_JT_CASE(5);
                __DPL_JT_CASE(6);
                __DPL_JT_DEFAULT(7);
            }
        } else if constexpr (size == 6 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_CASE(1);
                __DPL_JT_CASE(2);
                __DPL_JT_CASE(3);
                __DPL_JT_CASE(4);
                __DPL_JT_CASE(5);
                __DPL_JT_DEFAULT(6);
            }
        } else if constexpr (size == 5 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_CASE(1);
                __DPL_JT_CASE(2);
                __DPL_JT_CASE(3);
                __DPL_JT_CASE(4);
                __DPL_JT_DEFAULT(5);
            }
        } else if constexpr (size == 4 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_CASE(1);
                __DPL_JT_CASE(2);
                __DPL_JT_CASE(3);
                __DPL_JT_DEFAULT(4);
            }
        } else if constexpr (size == 3 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_CASE(1);
                __DPL_JT_CASE(2);
                __DPL_JT_DEFAULT(3);
            }
        } else if constexpr (size == 2 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_CASE(1);
                __DPL_JT_DEFAULT(2);
            }
        } else if constexpr (size == 1 + I) {
            switch (value) {
                __DPL_JT_CASE(0);
                __DPL_JT_DEFAULT(1);
            }
        }

        DPL_BUILTIN_unreachable();
#undef __DPL_JT_CASE
#undef __DPL_JT_DEFAULT
    }

public:
    template <T N0, T... Others>
    static constexpr jump_table<T, Ns..., N0, Others...> add_case() noexcept {
        return {};
    }

    // if return type is not void, must have a default_case
    // if return type void, default_case is optional
    template <typename F, dpl::convertible_to<T> U, typename... Args>
    requires requires { typename result_type<F, Args...>; }
    static constexpr decltype(auto) operator()(
        F&& callable, U&& value, Args&&... args) {
        return impl<0>(
            _JT_FWD(callable), static_cast<T>(value), _JT_FWD(args)...);
    }
};

#undef _JT_FWD
} // namespace mpfr
