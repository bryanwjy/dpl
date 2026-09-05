// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/equivalence.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename T, typename L>
concept vrot_vector_for =
    equivalent_vector_with<L, unsigned_canonical_vector_t<T>>;

} // namespace datapar::internal
__DPL_DEFAULT_NAMESPACE_END
