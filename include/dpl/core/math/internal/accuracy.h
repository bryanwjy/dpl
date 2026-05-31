// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/math/internal/fwd.h" // IWYU pragma: export

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_base_of.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {
struct accuracy_t {
    __DPL_HIDE_FROM_ABI constexpr ~accuracy_t() = default;
};

template <typename T>
concept accuracy_tag = is_base_of_v<fmath::accuracy_t, T>;

namespace accuracy {
struct maximum_t : private fmath::accuracy_t {
    __DPL_HIDE_FROM_ABI explicit constexpr maximum_t() noexcept = default;
};

struct speed_t : private maximum_t {
    __DPL_HIDE_FROM_ABI explicit constexpr speed_t() noexcept = default;
};

inline constexpr maximum_t maximum{};
inline constexpr speed_t speed{};
} // namespace accuracy
} // namespace datapar::fmath
DPL_DEFAULT_NAMESPACE_END
