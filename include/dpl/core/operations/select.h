// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate_mask.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/invocable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
void select(...) noexcept = delete;
template <auto>
void select(...) noexcept = delete;

struct select_t {

public:
    template <simd_class T, simd_class F, immediate_mask_for<T> M>
    static constexpr auto operator()(M mask, T tval, F val) noexcept = delete;
};

template <auto V>
struct selecti_t {};
template <integral auto V>
struct selecti_t<V> {
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
