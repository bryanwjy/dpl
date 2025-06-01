#pragma once

#include "dpl/config.h"

#include "dpl/concepts.h"

#if !DPL_MODULES
#  include <concepts>
#  include <type_traits>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace expressions {

namespace details::evaluate {

template <typename T>
__DPL_HIDE_FROM_ABI void evaluate(T&&) noexcept = delete;

class eval_t;

template <typename T>
concept evalutable_result = requires(eval_t eval, T result) { eval(result); };

template <typename T>
concept has_member_types = requires {
    // May be able to add more constraints for more complex expressions, e.g.
    // multi types?
    typename T::element_type;
    typename T::native_type;
};

class eval_t {
public:
    template <has_member_types T>
    requires requires(T const& expr) {
        { expr.evalute() } -> simd_type;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static inline auto operator()(
        T const& expr) const noexcept {
        return expr.evaluate();
    }

    template <has_member_types T>
    requires requires(T const& expr) {
        { evaluate(expr) } -> simd_type;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static inline auto operator()(
        T const& expr) const noexcept {
        return evaluate(expr);
    }

    template <has_member_types T>
    requires requires(T const& expr) {
        { expr.evaluate() } -> evalutable_result;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static inline auto operator()(
        T const& expr) const noexcept {
        return expr.evaluate();
    }

    template <has_member_types T>
    requires requires(T const& expr) {
        { evaluate(expr) } -> evalutable_result;
    }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static inline auto operator()(
        T const& expr) const noexcept {
        return evaluate(expr);
    }
};

DPL_EXPORT_BEGIN

inline namespace cpo {
inline constexpr details::evaluate::eval_t evaluate{};
}

template <typename T>
concept expression = requires(T const& expr) { evaluate(expr); };

template <typename T>
using expression_result_t = decltype(evaluate(std::declval<T>()));

template <typename T>
concept foldable_expression =
    expression<T> && expression<expression_result_t<T>>;

concept value_expression = expression<T> && simd_type<expression_result_t<T>>;

template <expression T>
using expression_element_t = typename T::element_type;

template <expression T>
using expression_native_type_t = typename T::native_type;

DPL_EXPORT_END

namespace details::evaluate {

class fold_t {
    static constexpr eval_t base_evaluator{};

public:
    template <foldable_expression T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static auto operator()(
        T const& expr) const noexcept {
        return operator()(base_evaluator(expr));
    }

    template <expression T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static auto operator()(
        T const& expr) const noexcept {
        return base_evaluator(expr);
    }

    template <simd_type T>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) static auto operator()(
        T const& value) const noexcept {
        return value;
    }
};

} // namespace details::evaluate

DPL_EXPORT_BEGIN

inline namespace cpo {
inline constexpr details::evaluate::fold_t fold{};
}

template <typename T>
using fold_result_t = decltype(fold(std::declval<T>()));

class __DPL_ABI_PUBLIC expression_interface {
public:
    template <expression Self>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline
    operator fold_result_t<Self>(this Self const& self) noexcept {
        return fold(self);
    }

    template <expression Self>
    DPL_ATTRIBUTES(
        _HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline fold_result_t<Self>
    operator+(this Self const& self) noexcept {
        return fold(self);
    }

protected:
    __DPL_HIDE_FROM_ABI inline constexpr ~expression_interface() noexcept =
        default;
};

DPL_EXPORT_END

template <typename T, typename U>
concept expression_result_with = expression<T> && expression<U> &&
    std::same_as<fold_result_t<T>, fold_result_t<U>>;

template <typename T, typename U>
concept simd_type_with_expression =
    simd_type<T> && expression<U> && std::same_as<fold_result_t<U>, T>;

} // namespace expressions

DPL_EXPORT_BEGIN

#define __DPL_FALLBACK_EXPRESSION_OPERATOR(SYMBOL)                          \
    template <expressions::expression L,                                    \
        expressions::expression_result_with<L> R>                           \
    requires requires(expressions::fold_result_t<L> left,                   \
                 expressions::fold_result_t<R> right) {                     \
        left SYMBOL right;                                                  \
    } &&                                                                    \
        (                                                                   \
            !requires(expressions::fold_result_t<L> left, R const& right) { \
                left SYMBOL right;                                          \
            } &&                                                            \
            !requires(L const& left, expressions::fold_result_t<R> right) { \
                left SYMBOL right;                                          \
            })                                                              \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) inline auto operator SYMBOL(  \
        L const& left, R const& right) noexcept {                           \
        return fold(left) SYMBOL fold(right);                               \
    }                                                                       \
    template <expressions::expression L,                                    \
        expressions::expression_result_with<L> R>                           \
    requires requires(expressions::fold_result_t<L> left, R const& right) { \
        left SYMBOL right;                                                  \
    } && (!requires(L const& left, expressions::fold_result_t<R> right) {   \
        left SYMBOL right;                                                  \
    })                                                                      \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) inline auto operator SYMBOL(  \
        L const& left, R const& right) noexcept {                           \
        return fold(left) SYMBOL right;                                     \
    }                                                                       \
    template <expressions::expression L,                                    \
        expressions::expression_result_with<L> R>                           \
    requires requires(L const& left, expressions::fold_result_t<R> right) { \
        left SYMBOL right;                                                  \
    }                                                                       \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) inline auto operator SYMBOL(  \
        L left, R right) noexcept {                                         \
        return left SYMBOL fold(right);                                     \
    }                                                                       \
    template <expressions::expression L,                                    \
        expressions::simd_type_with_expression<L> R>                        \
    requires requires(expressions::fold_result_t<L> left, R const& right) { \
        left SYMBOL right;                                                  \
    } && (!requires(L const& left, R const& right) { left SYMBOL right; })  \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) inline auto operator SYMBOL(  \
        L const& left, R const& right) noexcept {                           \
        return fold(left) SYMBOL right;                                     \
    }                                                                       \
    template <expressions::expression R,                                    \
        expressions::simd_type_with_expression<R> L>                        \
    requires requires(L const& left, expressions::fold_result_t<R> right) { \
        left SYMBOL right;                                                  \
    }                                                                       \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) inline auto operator SYMBOL(  \
        L const& left, R const& right) noexcept {                           \
        return left SYMBOL fold(right);                                     \
    }                                                                       \
    static_assert(true)

__DPL_FALLBACK_EXPRESSION_OPERATOR(+);
__DPL_FALLBACK_EXPRESSION_OPERATOR(-);
__DPL_FALLBACK_EXPRESSION_OPERATOR(/);
__DPL_FALLBACK_EXPRESSION_OPERATOR(*);
__DPL_FALLBACK_EXPRESSION_OPERATOR(&);
__DPL_FALLBACK_EXPRESSION_OPERATOR(|);
__DPL_FALLBACK_EXPRESSION_OPERATOR(^);
__DPL_FALLBACK_EXPRESSION_OPERATOR(<);
__DPL_FALLBACK_EXPRESSION_OPERATOR(>);
__DPL_FALLBACK_EXPRESSION_OPERATOR(<=);
__DPL_FALLBACK_EXPRESSION_OPERATOR(>=);
__DPL_FALLBACK_EXPRESSION_OPERATOR(==);
__DPL_FALLBACK_EXPRESSION_OPERATOR(!=);
__DPL_FALLBACK_EXPRESSION_OPERATOR(<<);
__DPL_FALLBACK_EXPRESSION_OPERATOR(>>);

#undef __DPL_FALLBACK_EXPRESSION_OPERATOR

DPL_EXPORT_END

DPL_DEFAULT_NAMESPACE_END
