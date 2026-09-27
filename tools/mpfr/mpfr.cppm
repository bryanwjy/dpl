// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <mpfr.h>
#include <string.h>

export module mpfr;
import dpl;
export import :mapped_memory;
export import :jump_table;
export import :utils;
export import :functions;
export import :cli;
export import :types;

export namespace mpfr {

class raw_value {
public:
    inline explicit raw_value(unsigned precision) noexcept {
        mpfr_init2(data_, precision);
    }

    inline ~raw_value() noexcept { destroy(); }

    inline raw_value(raw_value const& other)
        : raw_value(mpfr_get_prec(+other)) {
        mpfr_set(data_, +other, MPFR_RNDN);
    }

    inline raw_value& operator=(raw_value const& other) {
        if (this != &other) {
            auto const digits = mpfr_get_prec(+other);
            if (digits != mpfr_get_prec(data_)) {
                destroy();
                mpfr_init2(data_, digits);
            }

            mpfr_set(data_, +other, MPFR_RNDN);
        }

        return *this;
    }

    inline raw_value(raw_value&& other) {
        memcpy(data_, other.data_, sizeof(data_));
        memset(other.data_, 0, sizeof(data_));
    }

    inline raw_value& operator=(raw_value&& other) {
        destroy();
        memcpy(data_, other.data_, sizeof(data_));
        memset(other.data_, 0, sizeof(other.data_));
        return *this;
    }

    inline mpfr_ptr get() noexcept { return data_; }
    inline mpfr_srcptr get() const noexcept { return data_; }
    inline mpfr_ptr operator+() noexcept { return data_; }
    inline mpfr_srcptr operator+() const noexcept { return data_; }

    inline explicit operator float() const noexcept {
        return mpfr_get_flt(data_, MPFR_RNDN);
    }
    inline explicit operator double() const noexcept {
        return mpfr_get_d(data_, MPFR_RNDN);
    }

    inline explicit operator dpl::ext::float16() const noexcept {
        return mpfr_get_flt(data_, MPFR_RNDN);
    }

    inline explicit operator dpl::ext::bfloat16() const noexcept {
        return mpfr_get_flt(data_, MPFR_RNDN);
    }

private:
    void destroy() noexcept {
        constexpr mpfr_t null{};
        if (memcmp(data_, null, sizeof(null)) != 0) {
            mpfr_clear(data_);
        }
    }

    mpfr_t data_;
};

struct from_floating_point_t {
    explicit constexpr from_floating_point_t() noexcept = default;
};
inline constexpr from_floating_point_t from_floating_point{};
struct from_generator_t {
    explicit constexpr from_generator_t() noexcept = default;
};
inline constexpr from_generator_t from_generator{};

template <unsigned P>
class value : public raw_value {
public:
    inline explicit value() noexcept : raw_value(P) {}

    template <dpl::floating_point T>
    inline value(T arg) noexcept
    requires (P == dpl::floating_point_traits<T>::digits)
        : value(from_floating_point, arg) {}

    inline explicit value(from_floating_point_t, float arg) noexcept
    requires (P >= dpl::floating_point_traits<float>::digits)
        : value() {
        mpfr_set_flt(this->get(), arg, MPFR_RNDN);
    }
    inline explicit value(from_floating_point_t, double arg) noexcept
    requires (P >= dpl::floating_point_traits<double>::digits)
        : value() {
        mpfr_set_d(this->get(), arg, MPFR_RNDN);
    }
    inline explicit value(from_floating_point_t, long double arg) noexcept
    requires (P >= dpl::floating_point_traits<long double>::digits)
        : value() {
        mpfr_set_ld(this->get(), arg, MPFR_RNDN);
    }

    inline value(value const&) = default;
    inline value& operator=(value const&) = default;
    inline value(value&&) = default;
    inline value& operator=(value&&) = default;
};

template <unsigned P>
class result : public value<P> {
    template <unsigned>
    friend class result;

public:
    inline explicit result() noexcept = default;
    inline explicit result(result const&) noexcept = default;
    inline explicit result(result&&) noexcept = default;

    template <dpl::floating_point_like T>
    inline result(T arg) noexcept
    requires (P == dpl::floating_point_traits<T>::digits)
        : result(from_floating_point, arg) {}

    inline result(value<P> const& arg) noexcept
        : result(from_floating_point, arg) {}

    inline explicit result(from_floating_point_t, float arg) noexcept
        : result() {
        ternary = mpfr_set_flt(this->get(), arg, MPFR_RNDN);
    }

    inline explicit result(from_floating_point_t, double arg) noexcept
        : result() {
        ternary = mpfr_set_d(this->get(), arg, MPFR_RNDN);
    }
    inline explicit result(from_floating_point_t, long double arg) noexcept
        : result() {
        ternary = mpfr_set_ld(this->get(), arg, MPFR_RNDN);
    }
    inline explicit result(from_floating_point_t, raw_value const& arg) noexcept
        : result() {
        ternary = mpfr_set(this->get(), arg.get(), MPFR_RNDN);
    }
    template <unsigned P2>
    inline explicit result(
        from_floating_point_t tag, result<P2> const& other) noexcept
        : result(tag, static_cast<raw_value const&>(other)) {
        if (ternary == 0) {
            ternary = other.ternary;
        }
    }

    template <typename... Args, dpl::invocable<mpfr_ptr, Args...> F>
    requires dpl::same_as<dpl::invoke_result_t<F, mpfr_ptr, Args...>, int>
    inline explicit result(from_generator_t, F&& func, Args&&... args) noexcept
        : result() {
        ternary = dpl::invoke(
            dpl::forward<F>(func), this->get(), dpl::forward<Args>(args)...);
    }

    template <typename F, typename... Args>
    inline void update(F&& func, Args&&... args) noexcept {
        ternary = dpl::invoke(
            dpl::forward<F>(func), this->get(), dpl::forward<Args>(args)...);
    }

    template <unsigned P2>
    friend bool is_exact(result<P2> const&) noexcept;

    inline result& operator=(float arg) noexcept {
        ternary = mpfr_set_flt(this->get(), arg, MPFR_RNDN);
    }
    inline result& operator=(double arg) noexcept {
        ternary = mpfr_set_d(this->get(), arg, MPFR_RNDN);
    }
    inline result& operator=(long double arg) noexcept {
        ternary = mpfr_set_ld(this->get(), arg, MPFR_RNDN);
    }
    inline result& operator=(raw_value const& arg) noexcept {
        ternary = mpfr_set(this->get(), arg.get(), MPFR_RNDN);
    }

    template <dpl::floating_point_like T, unsigned P2>
    friend result<dpl::floating_point_traits<T>::digits> result_cast(
        result<P2>&& prev) noexcept;
    template <dpl::floating_point_like T, unsigned P2>
    friend result<dpl::floating_point_traits<T>::digits> result_cast(
        result<P2> const& prev) noexcept;

private:
    inline void check_range() noexcept {
        ternary = mpfr_check_range(this->get(), ternary, MPFR_RNDN);
    }

    inline void subnormalize() noexcept {
        ternary = mpfr_subnormalize(this->get(), ternary, MPFR_RNDN);
    }

    int ternary = 0;
};

class exponent_range_guard {
public:
    inline exponent_range_guard(mpfr_exp_t emin, mpfr_exp_t emax)
        : emin_(mpfr_get_emin())
        , emax_(mpfr_get_emax()) {
        mpfr_set_emin(emin);
        mpfr_set_emax(emax);
    }
    inline ~exponent_range_guard() {
        mpfr_set_emin(emin_);
        mpfr_set_emax(emax_);
    }
    inline exponent_range_guard(exponent_range_guard const&) = delete;
    inline exponent_range_guard& operator=(
        exponent_range_guard const&) = delete;

private:
    mpfr_exp_t emin_, emax_;
};

template <unsigned P2>
inline bool is_exact(result<P2> const& result) noexcept {
    return result.ternary == 0;
}

inline bool is_regular(raw_value const& val) noexcept {
    return mpfr_regular_p(+val);
}

inline bool is_number(raw_value const& val) noexcept {
    return mpfr_number_p(+val);
}

inline bool is_nan(raw_value const& val) noexcept {
    return mpfr_nan_p(+val);
}

inline bool is_inf(raw_value const& val) noexcept {
    return mpfr_inf_p(+val);
}

inline bool is_zero(raw_value const& val) noexcept {
    return mpfr_zero_p(+val);
}

inline auto get_exp(raw_value const& val) noexcept {
    return mpfr_get_exp(+val);
}

template <dpl::floating_point_like T, unsigned P>
inline result<dpl::floating_point_traits<T>::digits> result_cast(
    result<P> const& prev) noexcept {
    return result_cast<T>(result<P>(prev));
}

template <dpl::floating_point_like T, unsigned P>
inline result<dpl::floating_point_traits<T>::digits> result_cast(
    result<P>&& prev) noexcept {
    if (!is_exact(prev) && is_regular(prev) && mpfr_min_prec(+prev) < P) {
        if (prev.ternary > 0) {
            mpfr_nextbelow(+prev);
        } else {
            mpfr_nextabove(+prev);
        }
    }

    constexpr auto const min = 2 -
        dpl::floating_point_traits<T>::exponent_bias -
        dpl::floating_point_traits<T>::digits + 1;
    constexpr auto const max = dpl::floating_point_traits<T>::exponent_bias + 1;
    using return_type = result<dpl::floating_point_traits<T>::digits>;
    return_type ret(mpfr::from_floating_point, prev);
    exponent_range_guard const _(min, max);
    ret.check_range();
    ret.subnormalize();
    return ret;
}

} // namespace mpfr
