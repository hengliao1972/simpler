# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Single-source Python build constants for the A5 FDWIC runtime."""

FDWIC_TENSORMAP_RING_CAP_DEFINITION = "PTO_FDWIC_TENSORMAP_RING_CAP"
FDWIC_TENSORMAP_RING_CAP = 128

FDWIC_SCHEDULER_MODE_ENV = "PTO_FDWIC_SCHEDULER_MODE"
FDWIC_SCHEDULER_MODE_DEFINITION = "PTO_FDWIC_SCHEDULER_MODE"
FDWIC_SCHEDULER_MODE_SAME_CORE = "same_core"
FDWIC_SCHEDULER_MODES = (
    FDWIC_SCHEDULER_MODE_SAME_CORE,
    "cross_core_ordinary",
    "cross_core_dag",
    "simt_cross_core_ordinary",
    "simt_cross_core_dag",
)
FDWIC_SCHEDULER_MODE_IDS = {mode: index for index, mode in enumerate(FDWIC_SCHEDULER_MODES)}
FDWIC_SIMT_ORDINARY_BUILDER_LIMIT = 1
FDWIC_SIMT_DAG_BUILDER_LIMIT = 16


def fdwic_tensormap_ring_cap_definition() -> str:
    """Return the compile definition shared by all FDWIC artifact families."""

    return f"{FDWIC_TENSORMAP_RING_CAP_DEFINITION}={FDWIC_TENSORMAP_RING_CAP}"


def normalize_fdwic_scheduler_mode(mode: str) -> str:
    """Validate one public scheduler name and return its canonical spelling."""

    if not isinstance(mode, str) or mode.strip().lower() not in FDWIC_SCHEDULER_MODE_IDS:
        choices = ", ".join(FDWIC_SCHEDULER_MODES)
        raise ValueError(f"Invalid FDWIC scheduler mode {mode!r}; expected one of: {choices}")
    return mode.strip().lower()


def fdwic_scheduler_mode_definition(mode: str) -> str:
    """Return the numeric compile definition for one scheduler backend."""

    normalized = normalize_fdwic_scheduler_mode(mode)
    return f"{FDWIC_SCHEDULER_MODE_DEFINITION}={FDWIC_SCHEDULER_MODE_IDS[normalized]}"


def fdwic_simt_builder_limit(mode: str) -> int:
    """Return the compiled builder-block limit for one scheduler backend."""

    normalized = normalize_fdwic_scheduler_mode(mode)
    if normalized == "simt_cross_core_dag":
        return FDWIC_SIMT_DAG_BUILDER_LIMIT
    if normalized == "simt_cross_core_ordinary":
        return FDWIC_SIMT_ORDINARY_BUILDER_LIMIT
    return 0


def fdwic_simt_builder_count(mode: str, block_count: int) -> int:
    """Bound one scheduler's builder population by the physical block count."""

    if not isinstance(block_count, int) or isinstance(block_count, bool) or block_count < 0:
        raise ValueError(f"Invalid FDWIC block count {block_count!r}; expected a non-negative integer")
    return min(block_count, fdwic_simt_builder_limit(mode))
