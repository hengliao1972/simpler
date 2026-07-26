/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#pragma once

#include <cstdint>

// PTO_DIST_H 只接受非空 ASCII 十进制数字串，并完整解析到给定闭区间。
// 不借用 strtol 的空白/正负号语法，避免运维输入看似合法却改变保留窗口。
inline bool dist_parse_history_window(const char *text, int32_t max_inclusive, int32_t &value) {
    if (text == nullptr || *text == '\0' || max_inclusive < 0) return false;
    int32_t parsed = 0;
    for (const char *cursor = text; *cursor != '\0'; ++cursor) {
        if (*cursor < '0' || *cursor > '9') return false;
        const int32_t digit = static_cast<int32_t>(*cursor - '0');
        const int64_t next = static_cast<int64_t>(parsed) * 10 + digit;
        if (next > max_inclusive) return false;
        parsed = static_cast<int32_t>(next);
    }
    value = parsed;
    return true;
}
