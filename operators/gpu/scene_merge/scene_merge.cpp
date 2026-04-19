#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include <algorithm>
#include <cstdio>

// =============================================================================
// SceneMerge — N scene inputs → 1 combined scene output
// =============================================================================

/**
 * @brief Merges multiple 3D scene inputs into a single scene stream.
 *
 * SceneMerge is a structural utility for combining geometry, lights, and scene fragments before
 * they are passed into Render3D or downstream post-processing operators.
 */
struct SceneMerge : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "SceneMerge";
    static constexpr bool kTimeDependent = false;

    void collect_params(std::vector<vivid::ParamBase*>&) override {}

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(vivid::gpu::scene_port("scene_a", VIVID_PORT_INPUT));
        out.push_back(vivid::gpu::scene_port("scene_b", VIVID_PORT_INPUT));
        out.push_back(vivid::gpu::scene_port("scene_c", VIVID_PORT_INPUT));
        out.push_back(vivid::gpu::scene_port("scene_d", VIVID_PORT_INPUT));
        out.push_back(vivid::gpu::scene_port("scene",   VIVID_PORT_OUTPUT));
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
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
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 3.0f, "Merge");

        char badge[8];
        std::snprintf(badge, sizeof(badge), "%u/4", child_count_);
        float bw = d.text_width ? d.text_width(o, badge, 0.8f) : 20.0f;
        vivid::draw_plot::draw_thumb_value(d, o, w - bw - 6.0f, 3.0f, bw, badge);

        // Body: 4 input slots on the left, a short flow line, and an output
        // dot on the right. Filled slots represent active (non-null) inputs;
        // outlined slots represent empty ones. Slot identity (A/B/C/D) is
        // intentionally not shown because process_gpu compacts non-null inputs
        // so we can only count, not identify.
        float body_top = 18.0f;
        float body_h = h - body_top - 4.0f;
        float left_x = 8.0f;
        float right_x = w - 12.0f;

        // Slot column layout
        float slot_w = 14.0f;
        float slot_h = 8.0f;
        float col_h = slot_h * 4.0f + 3.0f * 4.0f; // 4 slots + 3 gaps
        float col_top = body_top + (body_h - col_h) * 0.5f;

        VividColor accent = {0.55f, 0.72f, 0.90f, 0.95f};
        VividColor dim    = {0.24f, 0.28f, 0.34f, 0.85f};

        for (int i = 0; i < 4; ++i) {
            float y = col_top + i * (slot_h + 3.0f);
            bool active = (static_cast<uint32_t>(i) < child_count_);
            VividColor c = active ? accent : dim;
            d.draw_rounded_rect(o, left_x, y, slot_w, slot_h, 2.0f, c);
        }

        // Merge chevrons: one line from each slot toward a single fan point.
        float fan_x = left_x + slot_w + (right_x - (left_x + slot_w)) * 0.55f;
        float fan_y = body_top + body_h * 0.5f;
        VividColor wire = {0.30f, 0.38f, 0.48f, 0.85f};
        VividColor active_wire = {0.70f, 0.85f, 1.0f, 0.85f};
        for (int i = 0; i < 4; ++i) {
            float slot_mid_y = col_top + i * (slot_h + 3.0f) + slot_h * 0.5f;
            VividColor c = (static_cast<uint32_t>(i) < child_count_) ? active_wire : wire;
            d.draw_line(o, left_x + slot_w + 1.0f, slot_mid_y, fan_x, fan_y, 1.0f, c);
        }

        // Output dot.
        d.draw_line(o, fan_x, fan_y, right_x - 4.0f, fan_y, 1.2f, active_wire);
        float dot = 3.0f;
        d.draw_rounded_rect(o, right_x - dot * 2.0f, fan_y - dot,
                            dot * 2.0f, dot * 2.0f, dot,
                            VividColor{1.0f, 0.78f, 0.31f, 0.95f});
    }

    void process_gpu(const VividGpuContext* ctx) override {
        // Collect non-null scene inputs
        child_count_ = 0;
        for (uint32_t i = 0; i < ctx->custom_input_count && child_count_ < 4; ++i) {
            auto* s = vivid::gpu::scene_input(ctx, i);
            if (s) {
                children_[child_count_++] = s;
            }
        }

        if (child_count_ == 0) return;

        // Output fragment: identity transform, no geometry, children = collected inputs
        vivid::gpu::scene_fragment_identity(output_);
        output_.vertex_buffer   = nullptr;
        output_.vertex_buf_size = 0;
        output_.index_buffer    = nullptr;
        output_.index_count     = 0;
        output_.pipeline        = nullptr;
        output_.material_binds  = nullptr;
        output_.fragment_type   = vivid::gpu::VividSceneFragment::GEOMETRY;
        output_.children        = children_;
        output_.child_count     = child_count_;

        ctx->custom_outputs[0] = &output_;
    }

private:
    vivid::gpu::VividSceneFragment  output_{};
    vivid::gpu::VividSceneFragment* children_[4]{};
    uint32_t                        child_count_ = 0;
};

VIVID_REGISTER(SceneMerge)
VIVID_THUMBNAIL(SceneMerge)

VIVID_DESCRIBE_REF_TYPE(vivid::gpu::VividSceneFragment)
