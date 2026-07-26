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

#include <gtest/gtest.h>

#include <cstdint>

#include "dist_engine/aicpu/config_parse.h"

namespace {

TEST(FdwicConfigParse, AcceptsOnlyInRangeAsciiDecimalDigitStrings) {
    struct ValidCase {
        const char *text;
        int32_t max_inclusive;
        int32_t expected;
    };
    constexpr ValidCase cases[] = {
        {"0", 0, 0},
        {"00", 1022, 0},
        {"64", 1022, 64},
        {"00064", 1022, 64},
        {"1022", 1022, 1022},
    };

    for (const ValidCase &test : cases) {
        int32_t value = -1;
        EXPECT_TRUE(dist_parse_history_window(test.text, test.max_inclusive, value)) << test.text;
        EXPECT_EQ(value, test.expected) << test.text;
    }
}

TEST(FdwicConfigParse, RejectsNullEmptyWhitespaceSignsAndNonDigitsWithoutChangingOutput) {
    constexpr const char *invalid[] = {
        "",
        " ",
        " 64",
        "64 ",
        "\t64",
        "64\n",
        "+64",
        "-1",
        "64x",
        "x64",
        "6_4",
    };

    int32_t value = 777;
    EXPECT_FALSE(dist_parse_history_window(nullptr, 1022, value));
    EXPECT_EQ(value, 777);

    for (const char *text : invalid) {
        value = 777;
        EXPECT_FALSE(dist_parse_history_window(text, 1022, value)) << text;
        EXPECT_EQ(value, 777) << text;
    }
}

TEST(FdwicConfigParse, RejectsOverflowAndValuesOutsideTheConfiguredRange) {
    constexpr const char *invalid[] = {
        "1023",
        "2147483648",
        "999999999999999999999999999999999999999999999999",
    };

    for (const char *text : invalid) {
        int32_t value = 777;
        EXPECT_FALSE(dist_parse_history_window(text, 1022, value)) << text;
        EXPECT_EQ(value, 777) << text;
    }

    int32_t value = 777;
    EXPECT_FALSE(dist_parse_history_window("0", -1, value));
    EXPECT_EQ(value, 777);

    value = 777;
    EXPECT_FALSE(dist_parse_history_window("1", 0, value));
    EXPECT_EQ(value, 777);
}

}  // namespace
