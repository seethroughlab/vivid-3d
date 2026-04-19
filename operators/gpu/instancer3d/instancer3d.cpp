#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/thumbnail_instance_array.h"
#include "operator_api/instance_algorithms.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// =============================================================================
// Instancer3D Operator — renders one mesh N times with per-instance transforms
// =============================================================================

/**
 * @brief Generates instance transforms for repeated 3D layouts.
 *
 * Instancer3D creates patterned placements such as grids, circles, lines, and 3D lattices, then
 * hands those transforms to downstream instanced rendering operators.
 *
 * @param count Number of instances to generate.
 * @param layout Placement pattern.
 * @param spacing Distance between generated instances.
 * @param palette Built-in color palette for generated instances.
 */
struct Instancer3D : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "Instancer3D";
    static constexpr bool kTimeDependent = false;
    static constexpr VividLaneBehavior kLaneBehavior = VIVID_LANE_KERNEL;

    vivid::Param<int>   count   {"count",   16, 1, 4096};
    vivid::Param<int>   layout  {"layout",  0, {"Grid", "Circle", "Line", "Grid3D"}};
    vivid::Param<float> spacing {"spacing", 2.0f, 0.1f, 20.0f};
    vivid::Param<int>   palette {"palette", 0, {"None", "Warm", "Cool", "Neon"}};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        vivid::param_group(count, "Instancer");
        vivid::param_group(layout, "Instancer");
        vivid::param_group(spacing, "Instancer");
        vivid::param_group(palette, "Instancer");

        out.push_back(&count);
        out.push_back(&layout);
        out.push_back(&spacing);
        out.push_back(&palette);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_INPUT));              // 0
        // Legacy per-attribute lane-array inputs (kept for backward compat).
        // Prefer the unified `instances` input below for new graphs.
        out.push_back({"positions", VIVID_PORT_LANE_ARRAY,  VIVID_PORT_INPUT});   // 1
        out.push_back({"scales",    VIVID_PORT_LANE_ARRAY,  VIVID_PORT_INPUT});   // 2
        out.push_back({"colors",    VIVID_PORT_LANE_ARRAY,  VIVID_PORT_INPUT});   // 3
        out.push_back({"scale_x",   VIVID_PORT_LANE_ARRAY,  VIVID_PORT_INPUT});   // 4
        out.push_back({"scale_y",   VIVID_PORT_LANE_ARRAY,  VIVID_PORT_INPUT});   // 5
        out.push_back({"scale_z",   VIVID_PORT_LANE_ARRAY,  VIVID_PORT_INPUT});   // 6
        out.push_back({"rotations", VIVID_PORT_LANE_ARRAY,  VIVID_PORT_INPUT});   // 7
        // Unified per-instance data: one wire carrying N records of
        // {position, rotation, scale, color}. Supersedes the legacy lane-array
        // ports above when connected.
        out.push_back(VIVID_CUSTOM_REF_PORT("instances", VIVID_PORT_INPUT,
                                            vivid::gpu::InstanceArray3D));        // 8
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_OUTPUT));
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        static const char* kLayoutNames[] = { "Grid", "Circle", "Line", "Grid3D" };
        int li = layout.int_value();
        const char* label = (li >= 0 && li < 4) ? kLayoutNames[li] : "Grid";
        vivid::thumb_instances::draw_scatter(
            ctx, instances_.data(),
            static_cast<uint32_t>(instances_.size()),
            label);
    }

    void process_gpu(const VividGpuContext* ctx) override {

        // Check input scene
        if (ctx->custom_input_count == 0 || !vivid::gpu::scene_input(ctx, 0)) return;
        const auto* input = vivid::gpu::scene_input(ctx, 0);
        if (!input->vertex_buffer || input->index_count == 0) return;

        // Unified instances input — if connected, use it directly and skip the
        // legacy lane-array + layout-preset path entirely.
        const vivid::gpu::InstanceArray3D* bundle = nullptr;
        if (ctx->custom_input_count > 1 && ctx->custom_inputs && ctx->custom_inputs[1]) {
            bundle = static_cast<const vivid::gpu::InstanceArray3D*>(ctx->custom_inputs[1]);
        }
        if (bundle && bundle->data && bundle->count > 0) {
            uint32_t n = bundle->count;
            if (n > 4096) n = 4096;
            instances_.assign(bundle->data, bundle->data + n);

            uint32_t buf_size = n * sizeof(vivid::gpu::InstanceData3D);
            if (buf_size < 48) buf_size = 48;
            if (n != current_count_) {
                rebuild_storage(ctx, n, buf_size);
            }
            if (storage_buf_) {
                wgpuQueueWriteBuffer(ctx->queue, storage_buf_, 0,
                                     instances_.data(),
                                     n * sizeof(vivid::gpu::InstanceData3D));
            }

            fragment_ = *input;
            fragment_.instance_buffer = storage_buf_;
            fragment_.instance_count  = n;
            ctx->custom_outputs[0] = &fragment_;
            return;
        }

        // Legacy path — read lane-array inputs + apply layout preset.
        // Input port indices: scene=0, positions=1, scales=2, colors=3,
        //   scale_x=4, scale_y=5, scale_z=6, rotations=7, instances=8 (handled above)
        const float* pos_data = nullptr;
        uint32_t pos_len = 0;
        const float* scale_data = nullptr;
        uint32_t scale_len = 0;
        const float* color_data = nullptr;
        uint32_t color_len = 0;
        const float* sx_data = nullptr;
        uint32_t sx_len = 0;
        const float* sy_data = nullptr;
        uint32_t sy_len = 0;
        const float* sz_data = nullptr;
        uint32_t sz_len = 0;
        const float* rot_data = nullptr;
        uint32_t rot_len = 0;

        if (ctx->input_lanes) {
            if (ctx->input_lanes[1].length > 0) {
                pos_data = ctx->input_lanes[1].data;
                pos_len  = ctx->input_lanes[1].length;
            }
            if (ctx->input_lanes[2].length > 0) {
                scale_data = ctx->input_lanes[2].data;
                scale_len  = ctx->input_lanes[2].length;
            }
            if (ctx->input_lanes[3].length > 0) {
                color_data = ctx->input_lanes[3].data;
                color_len  = ctx->input_lanes[3].length;
            }
            if (ctx->input_lanes[4].length > 0) {
                sx_data = ctx->input_lanes[4].data;
                sx_len  = ctx->input_lanes[4].length;
            }
            if (ctx->input_lanes[5].length > 0) {
                sy_data = ctx->input_lanes[5].data;
                sy_len  = ctx->input_lanes[5].length;
            }
            if (ctx->input_lanes[6].length > 0) {
                sz_data = ctx->input_lanes[6].data;
                sz_len  = ctx->input_lanes[6].length;
            }
            if (ctx->input_lanes[7].length > 0) {
                rot_data = ctx->input_lanes[7].data;
                rot_len  = ctx->input_lanes[7].length;
            }
        }

        // Determine instance count: from positions lane (3 floats per instance) or param
        uint32_t n = static_cast<uint32_t>(count.int_value());
        if (pos_data && pos_len >= 3) {
            n = pos_len / 3;
        }
        if (n == 0) n = 1;
        if (n > 4096) n = 4096;

        // Build instance data
        instances_.resize(n);
        int layout_mode = layout.int_value();
        float sp = spacing.value;

        bool use_custom_positions = (pos_data && pos_len >= n * 3);
        if (use_custom_positions) {
            for (uint32_t i = 0; i < n; ++i) {
                instances_[i].position[0] = pos_data[i * 3 + 0];
                instances_[i].position[1] = pos_data[i * 3 + 1];
                instances_[i].position[2] = pos_data[i * 3 + 2];
            }
        } else {
            // Layout math lives in instance_algorithms.h.
            // 3D lays Grid/Circle/Line in the XZ plane (y-up floor); Grid3D is cubic.
            for (uint32_t i = 0; i < n; ++i) {
                switch (layout_mode) {
                    case 1: {
                        auto p = vivid::instancing::circle_2d(i, n, sp);
                        instances_[i].position[0] = p.x;
                        instances_[i].position[1] = 0.0f;
                        instances_[i].position[2] = p.y;
                        break;
                    }
                    case 2: {
                        auto p = vivid::instancing::line_2d(i, n, sp);
                        instances_[i].position[0] = p.x;
                        instances_[i].position[1] = 0.0f;
                        instances_[i].position[2] = 0.0f;
                        break;
                    }
                    case 3: {
                        auto p = vivid::instancing::grid_3d(i, n, sp);
                        instances_[i].position[0] = p.x;
                        instances_[i].position[1] = p.y;
                        instances_[i].position[2] = p.z;
                        break;
                    }
                    default: {
                        auto p = vivid::instancing::grid_2d(i, n, sp);
                        instances_[i].position[0] = p.x;
                        instances_[i].position[1] = 0.0f;
                        instances_[i].position[2] = p.y;
                        break;
                    }
                }
            }
        }

        // Scales: per-axis scale_x/y/z take priority, then uniform 'scales', then 1.0
        bool use_sx = (sx_data && sx_len > 0);
        bool use_sy = (sy_data && sy_len > 0);
        bool use_sz = (sz_data && sz_len > 0);
        bool use_custom_scales = (scale_data && scale_len >= n);
        for (uint32_t i = 0; i < n; ++i) {
            float uniform_s = use_custom_scales ? scale_data[i] : 1.0f;
            instances_[i].scale[0] = use_sx ? sx_data[i % sx_len] : uniform_s;
            instances_[i].scale[1] = use_sy ? sy_data[i % sy_len] : uniform_s;
            instances_[i].scale[2] = use_sz ? sz_data[i % sz_len] : uniform_s;
            instances_[i].rotation_x = 0.0f;
        }

        // Rotations (Y-axis, radians)
        bool use_rot = (rot_data && rot_len > 0);
        for (uint32_t i = 0; i < n; ++i) {
            instances_[i].rotation_y = use_rot ? rot_data[i % rot_len] : 0.0f;
        }

        // Colors: custom lane > palette > input material color
        bool use_custom_colors = (color_data && color_len >= n * 4);
        int pal = palette.int_value();
        for (uint32_t i = 0; i < n; ++i) {
            if (use_custom_colors) {
                instances_[i].color[0] = color_data[i * 4 + 0];
                instances_[i].color[1] = color_data[i * 4 + 1];
                instances_[i].color[2] = color_data[i * 4 + 2];
                instances_[i].color[3] = color_data[i * 4 + 3];
            } else if (pal > 0) {
                apply_palette(instances_[i].color, i, pal);
            } else {
                instances_[i].color[0] = input->color[0];
                instances_[i].color[1] = input->color[1];
                instances_[i].color[2] = input->color[2];
                instances_[i].color[3] = input->color[3];
            }
        }

        // Upload to storage buffer
        uint32_t buf_size = n * sizeof(vivid::gpu::InstanceData3D);
        if (buf_size < 48) buf_size = 48;

        if (n != current_count_) {
            rebuild_storage(ctx, n, buf_size);
        }
        if (storage_buf_) {
            wgpuQueueWriteBuffer(ctx->queue, storage_buf_, 0,
                                 instances_.data(), n * sizeof(vivid::gpu::InstanceData3D));
        }

        // Output: shallow copy of input fragment with instance data
        fragment_ = *input;
        fragment_.instance_buffer = storage_buf_;
        fragment_.instance_count  = n;

        ctx->custom_outputs[0] = &fragment_;
    }

    ~Instancer3D() override {
        vivid::gpu::release(storage_buf_);
    }

private:
    vivid::gpu::VividSceneFragment fragment_{};
    std::vector<vivid::gpu::InstanceData3D> instances_;
    WGPUBuffer storage_buf_   = nullptr;
    uint32_t   current_count_ = 0;

    static void apply_palette(float color[4], uint32_t index, int palette_id) {
        // 3-color palettes, cycled by instance index
        static constexpr float kWarm[3][3] = {
            {1.0f, 0.35f, 0.25f},  // coral
            {1.0f, 0.75f, 0.2f},   // gold
            {0.95f, 0.5f, 0.6f},   // rose
        };
        static constexpr float kCool[3][3] = {
            {0.2f, 0.6f, 1.0f},    // azure
            {0.3f, 0.9f, 0.7f},    // teal
            {0.55f, 0.35f, 1.0f},  // violet
        };
        static constexpr float kNeon[3][3] = {
            {1.0f, 0.1f, 0.5f},    // hot pink
            {0.1f, 1.0f, 0.6f},    // green
            {0.2f, 0.5f, 1.0f},    // blue
        };

        const float (*pal)[3] = kWarm;
        if (palette_id == 2) pal = kCool;
        else if (palette_id == 3) pal = kNeon;

        uint32_t ci = index % 3;
        color[0] = pal[ci][0];
        color[1] = pal[ci][1];
        color[2] = pal[ci][2];
        color[3] = 1.0f;
    }

    void rebuild_storage(const VividGpuContext* ctx, uint32_t count, uint32_t buf_size) {
        vivid::gpu::release(storage_buf_);
        current_count_ = count;

        if (count == 0) return;

        WGPUBufferDescriptor desc{};
        desc.label = vivid_sv("Instancer3D Storage");
        desc.size  = buf_size;
        desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        storage_buf_ = wgpuDeviceCreateBuffer(ctx->device, &desc);
    }
};

VIVID_REGISTER(Instancer3D)
VIVID_THUMBNAIL(Instancer3D)

VIVID_DESCRIBE_REF_TYPES2(vivid::gpu::VividSceneFragment, vivid::gpu::InstanceArray3D)
