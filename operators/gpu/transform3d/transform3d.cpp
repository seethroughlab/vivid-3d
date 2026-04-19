#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>

// =============================================================================
// Transform3D — scene-in, scene-out with TRS transform
// =============================================================================

/**
 * @brief Applies translation, rotation, and scale to incoming 3D geometry.
 *
 * Transform3D is the main spatial utility for repositioning meshes or scenes before rendering.
 *
 * @param pos_x Position offset along the X axis.
 * @param pos_y Position offset along the Y axis.
 * @param pos_z Position offset along the Z axis.
 * @param rot_y Rotation around the Y axis.
 * @param scale_x Scale along the X axis.
 */
struct Transform3D : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "Transform3D";
    static constexpr bool kTimeDependent = false;

    vivid::Param<float> pos_x   {"pos_x",   0.0f, -50.0f, 50.0f};
    vivid::Param<float> pos_y   {"pos_y",   0.0f, -50.0f, 50.0f};
    vivid::Param<float> pos_z   {"pos_z",   0.0f, -50.0f, 50.0f};
    vivid::Param<float> rot_x   {"rot_x",   0.0f, -6.283f, 6.283f};
    vivid::Param<float> rot_y   {"rot_y",   0.0f, -6.283f, 6.283f};
    vivid::Param<float> rot_z   {"rot_z",   0.0f, -6.283f, 6.283f};
    vivid::Param<float> scale_x {"scale_x", 1.0f,  0.01f, 50.0f};
    vivid::Param<float> scale_y {"scale_y", 1.0f,  0.01f, 50.0f};
    vivid::Param<float> scale_z {"scale_z", 1.0f,  0.01f, 50.0f};

    Transform3D() {
        vivid::semantic_tag(pos_x, "position_xyz");
        vivid::semantic_shape(pos_x, "scalar");
        vivid::semantic_intent(pos_x, "position_x");
        vivid::semantic_tag(pos_y, "position_xyz");
        vivid::semantic_shape(pos_y, "scalar");
        vivid::semantic_intent(pos_y, "position_y");
        vivid::semantic_tag(pos_z, "position_xyz");
        vivid::semantic_shape(pos_z, "scalar");
        vivid::semantic_intent(pos_z, "position_z");

        vivid::semantic_tag(rot_x, "rotation_radians");
        vivid::semantic_shape(rot_x, "scalar");
        vivid::semantic_unit(rot_x, "rad");
        vivid::semantic_intent(rot_x, "rotation_x");
        vivid::semantic_tag(rot_y, "rotation_radians");
        vivid::semantic_shape(rot_y, "scalar");
        vivid::semantic_unit(rot_y, "rad");
        vivid::semantic_intent(rot_y, "rotation_y");
        vivid::semantic_tag(rot_z, "rotation_radians");
        vivid::semantic_shape(rot_z, "scalar");
        vivid::semantic_unit(rot_z, "rad");
        vivid::semantic_intent(rot_z, "rotation_z");

        vivid::semantic_tag(scale_x, "scale_xyz");
        vivid::semantic_shape(scale_x, "scalar");
        vivid::semantic_intent(scale_x, "scale_x");
        vivid::semantic_tag(scale_y, "scale_xyz");
        vivid::semantic_shape(scale_y, "scalar");
        vivid::semantic_intent(scale_y, "scale_y");
        vivid::semantic_tag(scale_z, "scale_xyz");
        vivid::semantic_shape(scale_z, "scalar");
        vivid::semantic_intent(scale_z, "scale_z");
    }

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        vivid::param_group(pos_x, "Transform");
        vivid::param_group(pos_y, "Transform");
        vivid::param_group(pos_z, "Transform");
        vivid::param_group(rot_x, "Transform");
        vivid::param_group(rot_y, "Transform");
        vivid::param_group(rot_z, "Transform");
        vivid::param_group(scale_x, "Transform");
        vivid::param_group(scale_y, "Transform");
        vivid::param_group(scale_z, "Transform");

        out.push_back(&pos_x);
        out.push_back(&pos_y);
        out.push_back(&pos_z);
        out.push_back(&rot_x);
        out.push_back(&rot_y);
        out.push_back(&rot_z);
        out.push_back(&scale_x);
        out.push_back(&scale_y);
        out.push_back(&scale_z);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_INPUT));
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_OUTPUT));
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
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 3.0f, "Xform");

        // Badge surfaces which transform component is active. Priority R > S > T.
        const float eps_rot = 1e-3f;
        const float eps_scale = 1e-3f;
        const float eps_pos = 1e-3f;
        bool has_rot   = std::fabs(rot_x.value)  > eps_rot   ||
                         std::fabs(rot_y.value)  > eps_rot   ||
                         std::fabs(rot_z.value)  > eps_rot;
        bool has_scale = std::fabs(scale_x.value - 1.0f) > eps_scale ||
                         std::fabs(scale_y.value - 1.0f) > eps_scale ||
                         std::fabs(scale_z.value - 1.0f) > eps_scale;
        bool has_pos   = std::fabs(pos_x.value) > eps_pos ||
                         std::fabs(pos_y.value) > eps_pos ||
                         std::fabs(pos_z.value) > eps_pos;
        const char* badge = has_rot ? "R" : has_scale ? "S" : has_pos ? "T" : "id";
        float bw = d.text_width ? d.text_width(o, badge, 0.8f) : 12.0f;
        vivid::draw_plot::draw_thumb_value(d, o, w - bw - 6.0f, 3.0f, bw, badge);

        // Body area — draw a cube wireframe projected orthographically, rotated
        // and scaled by the operator's transform. Position nudges the centre a
        // little so translation is legible without dragging the cube offscreen.
        float body_top = 18.0f;
        float body_h = h - body_top - 4.0f;
        float body_w = w - 12.0f;
        float body_side = std::min(body_w, body_h);
        float cx = w * 0.5f;
        float cy = body_top + body_h * 0.5f;

        // Clamp scale to keep the cube visible but expressive. sqrt helps huge
        // scales still fit while preserving relative proportions.
        auto shrink = [](float s) {
            float sign = s < 0.0f ? -1.0f : 1.0f;
            return sign * std::sqrt(std::min(std::fabs(s), 10.0f));
        };
        float sx = shrink(scale_x.value);
        float sy = shrink(scale_y.value);
        float sz = shrink(scale_z.value);

        // Half-extent of the unit cube in screen space at scale=1. Leave a margin
        // so a unit cube with modest rotation doesn't touch the body edges.
        float base = body_side * 0.22f;

        float hx = base * sx;
        float hy = base * sy;
        float hz = base * sz;

        float rx = rot_x.value;
        float ry = rot_y.value;
        float rz = rot_z.value;
        float cx_r = std::cos(rx), sx_r = std::sin(rx);
        float cy_r = std::cos(ry), sy_r = std::sin(ry);
        float cz_r = std::cos(rz), sz_r = std::sin(rz);

        auto rotate = [&](float x, float y, float z, float& ox, float& oy, float& oz) {
            // X-rotation
            float y1 =  y * cx_r - z * sx_r;
            float z1 =  y * sx_r + z * cx_r;
            // Y-rotation
            float x2 =  x * cy_r + z1 * sy_r;
            float z2 = -x * sy_r + z1 * cy_r;
            // Z-rotation
            float x3 = x2 * cz_r - y1 * sz_r;
            float y3 = x2 * sz_r + y1 * cz_r;
            ox = x3; oy = y3; oz = z2;
        };

        // Translation: project pos (x, y) into a small shift within the body.
        auto shift = [body_side](float v) {
            float clamped = std::max(-1.0f, std::min(1.0f, v / 10.0f));
            return clamped * body_side * 0.12f;
        };
        float tx_scr = shift(pos_x.value);
        float ty_scr = -shift(pos_y.value); // +Y in world → up on screen

        float corners[8][3] = {
            {-hx, -hy, -hz}, { hx, -hy, -hz},
            { hx,  hy, -hz}, {-hx,  hy, -hz},
            {-hx, -hy,  hz}, { hx, -hy,  hz},
            { hx,  hy,  hz}, {-hx,  hy,  hz},
        };
        float screen[8][2];
        for (int i = 0; i < 8; ++i) {
            float rxf, ryf, rzf;
            rotate(corners[i][0], corners[i][1], corners[i][2], rxf, ryf, rzf);
            screen[i][0] = cx + rxf + tx_scr;
            screen[i][1] = cy - ryf + ty_scr;
        }

        static const int kEdges[12][2] = {
            {0,1},{1,2},{2,3},{3,0},         // back face
            {4,5},{5,6},{6,7},{7,4},         // front face
            {0,4},{1,5},{2,6},{3,7},         // connecting edges
        };
        VividColor line_col = {0.70f, 0.78f, 0.85f, 0.95f};
        for (auto& e : kEdges) {
            d.draw_line(o, screen[e[0]][0], screen[e[0]][1],
                        screen[e[1]][0], screen[e[1]][1], 1.2f, line_col);
        }

        // Pivot dot at projected origin.
        d.draw_rounded_rect(o, cx + tx_scr - 1.5f, cy + ty_scr - 1.5f,
                            3.0f, 3.0f, 1.5f,
                            VividColor{1.0f, 0.78f, 0.31f, 0.9f});
    }

    void process_gpu(const VividGpuContext* ctx) override {
        // No input scene → no output
        bool has_input = ctx->custom_input_count > 0 &&
                         vivid::gpu::scene_input(ctx, 0) != nullptr;
        if (!has_input) return;

        // Build TRS matrix: T * Rz * Ry * Rx * S (same order as Shape3D)
        mat4x4 S, tmp;
        mat4x4_identity(S);
        mat4x4_scale_aniso(S, S, scale_x.value, scale_y.value, scale_z.value);
        mat4x4_rotate_X(tmp, S, rot_x.value);
        mat4x4_rotate_Y(S, tmp, rot_y.value);
        mat4x4_rotate_Z(tmp, S, rot_z.value);

        mat4x4 T;
        mat4x4_translate(T, pos_x.value, pos_y.value, pos_z.value);
        mat4x4_mul(output_.model_matrix, T, tmp);

        // No geometry on this fragment — just a transform wrapper
        output_.vertex_buffer   = nullptr;
        output_.vertex_buf_size = 0;
        output_.index_buffer    = nullptr;
        output_.index_count     = 0;
        output_.pipeline        = nullptr;
        output_.material_binds  = nullptr;
        output_.fragment_type   = vivid::gpu::VividSceneFragment::GEOMETRY;

        // Wrap input as child
        child_ = vivid::gpu::scene_input(ctx, 0);
        output_.children    = &child_;
        output_.child_count = 1;

        ctx->custom_outputs[0] = &output_;
    }

private:
    vivid::gpu::VividSceneFragment  output_{};
    vivid::gpu::VividSceneFragment* child_ = nullptr;
};

VIVID_REGISTER(Transform3D)
VIVID_THUMBNAIL(Transform3D)

VIVID_DESCRIBE_REF_TYPE(vivid::gpu::VividSceneFragment)
