// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/basic_type.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::atom {
template <typename T, typename U>
concept common_basic_element_with = requires {
    typename basic_element_t<T>;
    typename basic_element_t<U>;
    requires same_as<basic_element_t<T>, basic_element_t<U>>;
};
} // namespace datapar::atom
DPL_DEFAULT_NAMESPACE_END
