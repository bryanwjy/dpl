// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/math/details/fpfix.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/type_traits/remove_cv.h"
#  include "dpl/std/utility/to_underlying.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT namespace datapar::fpfix {

inline constexpr mx::revert_t revert{};
inline constexpr mx::signed_inf_t signed_inf{};

template <typename T>
concept fpclass_type = mx::is_fpclass<T>;
template <typename T>
concept condition_set = mx::is_condition_set<T>;
template <typename T, typename E>
concept condition_set_for =
    floating_point<E> && condition_set<T> && mx::is_condition_set_for<T, E>;

template <typename T, typename E, auto... Rs>
concept result_subset_of =
    condition_set_for<T, E> && (... && mx::result_type<decltype(Rs), E>) &&
    mx::subset_of<decltype(Rs)...>(T{});

inline constexpr mx::template_for_t template_for{};

// clang-format off
inline constexpr mx::fpclass<mx::fpc::none> none{};
inline constexpr mx::fpclass<mx::fpc::zero> zero{};
inline constexpr mx::fpclass<mx::fpc::one> one{};
inline constexpr mx::fpclass<mx::fpc::positive> positive{};
inline constexpr mx::fpclass<mx::fpc::negative> negative{};
inline constexpr mx::fpclass<mx::fpc::pos_inf> pos_inf{};
inline constexpr mx::fpclass<mx::fpc::neg_inf> neg_inf{};
inline constexpr mx::fpclass<mx::fpc::qnan> qnan{};
inline constexpr mx::fpclass<mx::fpc::snan> snan{};
inline constexpr mx::fpclass<mx::fpc::all> all{};
inline constexpr mx::fpclass<mx::fpc::infinity> infinity{};
inline constexpr mx::fpclass<mx::fpc::nan> nan{};
inline constexpr mx::fpclass<mx::fpc::finite> finite{};
// clang-format on

template <mx::fpc C>
consteval int popcount(mx::fpclass<C>) noexcept {
    return __DPL popcount(__DPL to_underlying(C));
}

template <fpclass_type auto C, auto V>
inline constexpr mx::fixup_pair<static_cast<mx::fpc>(C),
    remove_cv_t<decltype(V)>>
    condition{};
} // namespace datapar::fpfix

DPL_EXPORT namespace fpfix = datapar::fpfix; // NOLINT

DPL_DEFAULT_NAMESPACE_END
