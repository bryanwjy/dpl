// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include "mpfr.h"

export module mpfr:functions;
import :utils;
import dpl;

export namespace mpfr {
template <size_t N>
struct function_name : static_string<N> {
private:
    using static_string<N>::data;

public:
    using static_string<N>::static_string;
};

template <size_t N>
function_name(char const (&data)[N]) -> function_name<N - 1>;

inline namespace literals {
template <function_name S>
consteval decltype(auto) operator""_func() noexcept {
    return (S);
}
} // namespace literals

template <uint64 V>
struct function_hash : dpl::integral_constant<uint64, V> {
    using dpl::integral_constant<uint64, V>::integral_constant;
    friend consteval decltype(auto) to_function_name(function_hash) noexcept;
    friend consteval auto to_function(function_hash) noexcept;

    friend auto invoke_from_hash(function_hash, mpfr_ptr, mpfr_srcptr) noexcept;
    friend auto invoke_from_hash(
        function_hash, mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t) noexcept;
    friend auto invoke_from_hash(
        function_hash, mpfr_ptr, mpfr_srcptr, mpfr_rnd_t) noexcept;
    friend auto invoke_from_hash(
        function_hash, mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t) noexcept;
};

namespace details {
template <typename F>
struct arity {};
template <typename R, typename... Args>
struct arity<R (*)(mpfr_ptr, Args...)> : dpl::size_constant<sizeof...(Args)> {};

// last argument is mpfr_rnd_t
template <typename R, typename... Args>
requires dpl::same_as<decltype((..., static_cast<Args*>(0))), mpfr_rnd_t*>
struct arity<R (*)(mpfr_ptr, Args...)> :
    dpl::size_constant<sizeof...(Args) - 1> {};

} // namespace details

void to_function(...) noexcept = delete;

template <mpfr::function_name N>
constexpr auto arity_v = details::arity<decltype(to_function(
    function_hash<mpfr::string_hash(N)>{}))>::value;

template <uint64 V>
requires requires(function_hash<V> hash) { to_function_name(hash); }
consteval decltype(auto) to_function_name(
    dpl::integral_constant<uint64, V> = {}) noexcept {
    constexpr function_hash<V> hash;
    return to_function_name(hash);
}

template <auto name, typename F, F func>
struct function_map_impl {};

template <auto name, typename R, typename... Args, R (*func)(Args...)>
struct function_map_impl<name, R (*)(Args...), func> {
    using hash_type = function_hash<string_hash(name)>;
    friend auto invoke_from_hash(hash_type, Args... args) noexcept {
        if constexpr (dpl::is_same_v<R, void>) {
            return 0;
        } else {
            return func(args...);
        }
    }
};

template <typename T>
struct remove_ptr {
    using type = T;
};
template <typename T>
struct remove_ptr<T*> {
    using type = T;
};

template <typename T>
using remove_ptr_t = typename remove_ptr<T>::type;

template <auto name, auto func>
struct function_map :
    function_map_impl<name, dpl::decay_t<decltype(func)>, func> {
    using hash_type = function_hash<string_hash(name)>;

    friend consteval decltype(auto) to_function_name(hash_type) noexcept {
        return (name);
    }

    friend consteval auto to_function(hash_type) noexcept { return func; }
};

void invoke_from_hash(...) noexcept = delete;

template <typename H, typename... Args>
concept invocable_from_hash =
    requires(H hash, Args... args) { invoke_from_hash(hash, args...); };

template <function_name name>
struct invoke_t {
    using hash_type = function_hash<string_hash(name)>;

    template <typename... Args>
    requires invocable_from_hash<hash_type, Args...>
    static auto operator()(Args... args) noexcept {
        constexpr hash_type hash;
        return invoke_from_hash(hash, args...);
    }

    static auto operator()(mpfr_ptr result, mpfr_srcptr arg, mpfr_srcptr exp,
        mpfr_rnd_t rnd) noexcept
    requires (!invocable_from_hash<hash_type, mpfr_ptr, mpfr_srcptr,
                 mpfr_srcptr, mpfr_rnd_t>) &&
        invocable_from_hash<hash_type, mpfr_ptr, mpfr_srcptr, long, mpfr_rnd_t>
    {
        constexpr hash_type hash;
        auto const lexp = mpfr_get_si(exp, MPFR_RNDN);
        return invoke_from_hash(hash, result, arg, lexp, rnd);
    }

    template <typename... Args>
    requires (!invocable_from_hash<hash_type, Args...>)
    static auto operator()(Args... args) noexcept
    requires requires { operator()(args..., MPFR_RNDN); }
    {
        return operator()(args..., MPFR_RNDN);
    }
};

template <function_name name>
inline constexpr invoke_t<name> invoke;

#define __MPFR_DEFINE_FUNCTION(NAME, FUNC)                          \
    [] {                                                            \
        constexpr auto name = DPL_CONCAT(#NAME, _func);             \
        static_assert(sizeof(mpfr::function_map<name, &FUNC>) > 0); \
        return name;                                                \
    }()

constexpr dpl::constant_type_pack<                       //
    __MPFR_DEFINE_FUNCTION(acos, mpfr_acos),             //
    __MPFR_DEFINE_FUNCTION(acosh, mpfr_acosh),           //
    __MPFR_DEFINE_FUNCTION(asin, mpfr_asin),             //
    __MPFR_DEFINE_FUNCTION(asinh, mpfr_asinh),           //
    __MPFR_DEFINE_FUNCTION(atan, mpfr_atan),             //
    __MPFR_DEFINE_FUNCTION(atan2, mpfr_atan2),           //
    __MPFR_DEFINE_FUNCTION(atanh, mpfr_atanh),           //
    __MPFR_DEFINE_FUNCTION(beta, mpfr_beta),             //
    __MPFR_DEFINE_FUNCTION(cbrt, mpfr_cbrt),             //
    __MPFR_DEFINE_FUNCTION(ceil, mpfr_ceil),             //
    __MPFR_DEFINE_FUNCTION(cos, mpfr_cos),               //
    __MPFR_DEFINE_FUNCTION(cosh, mpfr_cosh),             //
    __MPFR_DEFINE_FUNCTION(cot, mpfr_cot),               //
    __MPFR_DEFINE_FUNCTION(coth, mpfr_coth),             //
    __MPFR_DEFINE_FUNCTION(csc, mpfr_csc),               //
    __MPFR_DEFINE_FUNCTION(csch, mpfr_csch),             //
    __MPFR_DEFINE_FUNCTION(erf, mpfr_erf),               //
    __MPFR_DEFINE_FUNCTION(erfc, mpfr_erfc),             //
    __MPFR_DEFINE_FUNCTION(exp, mpfr_exp),               //
    __MPFR_DEFINE_FUNCTION(exp10, mpfr_exp10),           //
    __MPFR_DEFINE_FUNCTION(exp2, mpfr_exp2),             //
    __MPFR_DEFINE_FUNCTION(expm1, mpfr_expm1),           //
    __MPFR_DEFINE_FUNCTION(floor, mpfr_floor),           //
    __MPFR_DEFINE_FUNCTION(fmod, mpfr_fmod),             //
    __MPFR_DEFINE_FUNCTION(hypot, mpfr_hypot),           //
    __MPFR_DEFINE_FUNCTION(tgamma, mpfr_gamma),          //
    __MPFR_DEFINE_FUNCTION(lgamma, mpfr_lngamma),        //
    __MPFR_DEFINE_FUNCTION(log, mpfr_log),               //
    __MPFR_DEFINE_FUNCTION(log10, mpfr_log10),           //
    __MPFR_DEFINE_FUNCTION(log1p, mpfr_log1p),           //
    __MPFR_DEFINE_FUNCTION(log2, mpfr_log2),             //
    __MPFR_DEFINE_FUNCTION(nexttoward, mpfr_nexttoward), //
    __MPFR_DEFINE_FUNCTION(pow, mpfr_pow),               //
    __MPFR_DEFINE_FUNCTION(ldexp, mpfr_mul_2si),         //
    __MPFR_DEFINE_FUNCTION(remainder, mpfr_remainder),   //
    __MPFR_DEFINE_FUNCTION(round, mpfr_round),           // CMATH
    __MPFR_DEFINE_FUNCTION(roundeven, mpfr_roundeven),   // IEEE
    __MPFR_DEFINE_FUNCTION(sec, mpfr_sec),               //
    __MPFR_DEFINE_FUNCTION(sech, mpfr_sech),             //
    __MPFR_DEFINE_FUNCTION(sin, mpfr_sin),               //
    __MPFR_DEFINE_FUNCTION(sinh, mpfr_sinh),             //
    __MPFR_DEFINE_FUNCTION(sqrt, mpfr_sqrt),             //
    __MPFR_DEFINE_FUNCTION(tan, mpfr_tan),               //
    __MPFR_DEFINE_FUNCTION(tanh, mpfr_tanh),             //
    __MPFR_DEFINE_FUNCTION(trunc, mpfr_trunc),           //
    __MPFR_DEFINE_FUNCTION(zeta, mpfr_zeta)>
    functions = {};
#undef __MPFR_DEFINE_FUNCTION

} // namespace mpfr
