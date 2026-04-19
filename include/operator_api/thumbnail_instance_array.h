#pragma once

#include "operator_api/thumbnail.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/draw_plot_helpers.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace vivid::thumb_instances {

// Renders a 2D dot-scatter thumbnail showing an array of InstanceData3D records.
// Each instance becomes one dot at its (x, z) world position, sized by the mean
// scale and tinted by the instance color (falling back to the package default
// cool-blue when the color is uncolored/black).
//
// Layout:
//   [label]                          [count-badge]
//   (dot scatter filling the rest of the thumbnail)
//
// Preconditions:
//   - ctx->draw.opaque must be non-null (2D draw API available). The call is a
//     no-op otherwise.
inline void draw_scatter(const VividThumbnailContext* ctx,
                         const vivid::gpu::InstanceData3D* instances,
                         uint32_t count,
                         const char* label) {
    if (!ctx || !ctx->draw.opaque) return;
    VividDrawAPI d = ctx->draw;
    void* o = d.opaque;

    float w = static_cast<float>(ctx->thumbnail_logical_width
                                     ? ctx->thumbnail_logical_width
                                     : ctx->thumbnail_width);
    float h = static_cast<float>(ctx->thumbnail_logical_height
                                     ? ctx->thumbnail_logical_height
                                     : ctx->thumbnail_height);

    vivid::draw_plot::draw_thumb_background(d, o, w, h);

    if (label && label[0]) {
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 3.0f, label);
    }

    char badge[16];
    if (count >= 1000) std::snprintf(badge, sizeof(badge), "%uk", count / 1000);
    else               std::snprintf(badge, sizeof(badge), "%u", count);
    float bw = d.text_width ? d.text_width(o, badge, 0.8f) : 24.0f;
    vivid::draw_plot::draw_thumb_value(d, o, w - bw - 6.0f, 3.0f, bw, badge);

    if (!instances || count == 0) return;

    float xmin =  1e9f, xmax = -1e9f;
    float zmin =  1e9f, zmax = -1e9f;
    for (uint32_t i = 0; i < count; ++i) {
        float x = instances[i].position[0];
        float z = instances[i].position[2];
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
    }
    float xrange = std::max(1e-3f, xmax - xmin);
    float zrange = std::max(1e-3f, zmax - zmin);
    float range  = std::max(xrange, zrange);
    float xcenter = (xmin + xmax) * 0.5f;
    float zcenter = (zmin + zmax) * 0.5f;

    const float top_reserve = 14.0f;
    const float margin = 4.0f;
    float body_w = w - margin * 2.0f;
    float body_h = h - top_reserve - margin;
    float scale = std::min(body_w, body_h) / (range * 1.15f);
    float cx = margin + body_w * 0.5f;
    float cy = top_reserve + body_h * 0.5f;

    for (uint32_t i = 0; i < count; ++i) {
        const auto& inst = instances[i];
        float sx = cx + (inst.position[0] - xcenter) * scale;
        // Negate z so +Z reads as "up" in a top-down view.
        float sy = cy - (inst.position[2] - zcenter) * scale;

        float avg_scale = (inst.scale[0] + inst.scale[1] + inst.scale[2]) / 3.0f;
        float dot = std::clamp(avg_scale * 0.8f, 1.2f, 3.5f);

        float r = inst.color[0], g = inst.color[1], b = inst.color[2];
        float a = inst.color[3];
        // Uncolored-fallback: if the instance has no color set, use package default.
        if (r + g + b < 0.01f) { r = 0.65f; g = 0.72f; b = 0.80f; }
        if (a < 0.05f) a = 0.9f;

        d.draw_rect(o, sx - dot * 0.5f, sy - dot * 0.5f, dot, dot,
                    VividColor{r, g, b, a});
    }
}

} // namespace vivid::thumb_instances
