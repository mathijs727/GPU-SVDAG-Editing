#pragma once
#include "my_units.h"
#include <fmt/core.h>

template <>
struct fmt::formatter<my_units::bytes> {
    // Parses format specifications of the form ['f' | 'e'].
    constexpr auto parse(format_parse_context& ctx) -> format_parse_context::iterator
    {
        // No format string
        auto it = ctx.begin(), end = ctx.end();
        while (it != end && *it != '}')
            ++it;
        return it;
    }

    // Formats the point p using the parsed format specification (presentation)
    // stored in this formatter.
    auto format(const my_units::bytes& in, format_context& ctx) const -> format_context::iterator
    {
        /*const double bytes = in.convert<units::data::bytes>().value();
        const double kilobytes = in.convert<units::data::kilobytes>().value();
        const double megabytes = in.convert<units::data::megabytes>().value();
        const double gigabytes = in.convert<units::data::gigabytes>().value();*/

        const double bytes = (double)in;
        const double kilobytes = bytes / 1000.0;
        const double megabytes = kilobytes / 1000.0;
        const double gigabytes = megabytes / 1000.0;

        if (gigabytes > 1.0)
            return fmt::format_to(ctx.out(), "{:.2f}GB", gigabytes);
        else if (megabytes > 1.0)
            return fmt::format_to(ctx.out(), "{:.2f}MB", megabytes);
        else if (kilobytes > 1.0)
            return fmt::format_to(ctx.out(), "{:.2f}KB", kilobytes);
        else
            return fmt::format_to(ctx.out(), "{:.2f}B", bytes);
    }
};
