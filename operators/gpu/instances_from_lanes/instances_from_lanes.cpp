#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/thumbnail_instance_array.h"
#include <algorithm>
#include <cstdint>
#include <vector>

// =============================================================================
// InstancesFromLanes — pack per-attribute lane arrays into InstanceArray3D
// =============================================================================

/**
 * @brief Migration bridge from per-attribute lane arrays to an InstanceArray3D bundle.
 *
 * Accepts up to 11 optional lane-array inputs (position, scale, rotation, color
 * per component) and emits a single InstanceArray3D bundle for consumption by
 * Instancer3D's `instances` input. Use this when driving instance attributes
 * from independent lane sources (e.g. SpreadNoise, FFT analysis, Repeat).
 *
 * Instance count = max length among connected inputs. Unconnected attributes
 * fall back to sensible defaults: position/rotation 0, scale 1, color 1.
 */
struct InstancesFromLanes : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName         = "InstancesFromLanes";
    static constexpr bool kTimeDependent       = false;
    static constexpr VividLaneBehavior kLaneBehavior = VIVID_LANE_KERNEL;

    void collect_params(std::vector<vivid::ParamBase*>& /*out*/) override {}

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        // 11 optional lane-array inputs, fixed order:
        out.push_back({"pos_x",   VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 0
        out.push_back({"pos_y",   VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 1
        out.push_back({"pos_z",   VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 2
        out.push_back({"scale_x", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 3
        out.push_back({"scale_y", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 4
        out.push_back({"scale_z", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 5
        out.push_back({"rot_y",   VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 6
        out.push_back({"color_r", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 7
        out.push_back({"color_g", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 8
        out.push_back({"color_b", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 9
        out.push_back({"color_a", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});  // 10

        out.push_back(VIVID_CUSTOM_REF_PORT("instances", VIVID_PORT_OUTPUT,
                                            vivid::gpu::InstanceArray3D));
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        vivid::thumb_instances::draw_scatter(
            ctx, instances_.data(),
            static_cast<uint32_t>(instances_.size()),
            "Lanes");
    }

    void process_gpu(const VividGpuContext* ctx) override {
        // Collect pointers + lengths for each lane-array input (all optional).
        const float* in_data[11]{};
        uint32_t     in_len [11]{};

        if (ctx->input_lanes) {
            for (int p = 0; p < 11; ++p) {
                if (ctx->input_lanes[p].length > 0) {
                    in_data[p] = ctx->input_lanes[p].data;
                    in_len [p] = ctx->input_lanes[p].length;
                }
            }
        }

        // Count = max length among connected inputs (cap at 4096).
        uint32_t n = 0;
        for (int p = 0; p < 11; ++p) n = std::max(n, in_len[p]);
        if (n == 0) n = 1;
        if (n > 4096) n = 4096;

        instances_.resize(n);

        auto sample = [](const float* data, uint32_t len, uint32_t i, float fallback) -> float {
            return (data && len > 0) ? data[i % len] : fallback;
        };

        for (uint32_t i = 0; i < n; ++i) {
            auto& d = instances_[i];
            d.position[0] = sample(in_data[0], in_len[0], i, 0.0f);
            d.position[1] = sample(in_data[1], in_len[1], i, 0.0f);
            d.position[2] = sample(in_data[2], in_len[2], i, 0.0f);
            d.scale[0]    = sample(in_data[3], in_len[3], i, 1.0f);
            d.scale[1]    = sample(in_data[4], in_len[4], i, 1.0f);
            d.scale[2]    = sample(in_data[5], in_len[5], i, 1.0f);
            d.rotation_y  = sample(in_data[6], in_len[6], i, 0.0f);
            d.rotation_x  = 0.0f;
            d.color[0]    = sample(in_data[7], in_len[7], i, 1.0f);
            d.color[1]    = sample(in_data[8], in_len[8], i, 1.0f);
            d.color[2]    = sample(in_data[9], in_len[9], i, 1.0f);
            d.color[3]    = sample(in_data[10], in_len[10], i, 1.0f);
        }

        bundle_.data  = instances_.data();
        bundle_.count = n;
        ctx->custom_outputs[0] = &bundle_;
    }

private:
    std::vector<vivid::gpu::InstanceData3D> instances_;
    vivid::gpu::InstanceArray3D bundle_{};
};

VIVID_REGISTER(InstancesFromLanes)
VIVID_THUMBNAIL(InstancesFromLanes)

VIVID_DESCRIBE_REF_TYPE(vivid::gpu::InstanceArray3D)
