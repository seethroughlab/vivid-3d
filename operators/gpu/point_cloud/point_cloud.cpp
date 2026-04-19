#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_common.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/type_id.h"
#include "operator_api/port_type_registry.h"
#include "operator_api/thumbnail_instance_array.h"
#include <cstdio>
#include <cstring>
#include <vector>

// =============================================================================
// PointCloud — interprets a LANE_ARRAY as [x0,y0, x1,y1, ...] pairs
//              and produces a PointList mesh.
//
// Input:  "positions" (VIVID_PORT_LANE_ARRAY)
// Output: "mesh"      (VIVID_CUSTOM_PORT, topology PointList)
//
// Vertex layout: vec2f (xy) = 8 bytes per point.
// Rebuilds vertex buffer when point count changes; uploads lane data each tick.
// =============================================================================

/**
 * @brief Renders incoming 3D points as a shaded point cloud.
 *
 * PointCloud turns position data into a simple GPU point primitive, useful for raw scans,
 * particle visualizations, and lightweight spatial debugging.
 *
 * @param point_size Size of each rendered point.
 * @param r Red channel of the default point color.
 * @param g Green channel of the default point color.
 * @param b Blue channel of the default point color.
 */
struct PointCloud : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "PointCloud";
    static constexpr bool kTimeDependent = false;

    vivid::Param<float> point_size {"point_size", 0.01f, 0.001f, 0.1f};
    vivid::Param<float> r          {"r",          1.0f,  0.0f,   1.0f};
    vivid::Param<float> g          {"g",          1.0f,  0.0f,   1.0f};
    vivid::Param<float> b          {"b",          1.0f,  0.0f,   1.0f};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        out.push_back(&point_size);
        out.push_back(&r);
        out.push_back(&g);
        out.push_back(&b);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"positions", VIVID_PORT_LANE_ARRAY, VIVID_PORT_INPUT});
        out.push_back(VIVID_CUSTOM_REF_PORT("mesh", VIVID_PORT_OUTPUT, VividMesh));
    }

    void process_gpu(const VividGpuContext* ctx) override {
        if (ctx->custom_output_count == 0) return;

        // Read positions lane: pairs of [x,y]
        uint32_t lane_len = 0;
        const float* lane_data = nullptr;
        if (ctx->input_lanes && ctx->input_lanes[0].length > 0) {
            lane_len  = ctx->input_lanes[0].length;
            lane_data = ctx->input_lanes[0].data;
        }

        uint32_t point_count = lane_len / 2;

        if (point_count != built_count_) {
            rebuild(ctx, point_count);
        }

        // Upload current lane data each tick
        if (vertex_buf_ && lane_data && point_count > 0) {
            wgpuQueueWriteBuffer(ctx->queue, vertex_buf_, 0,
                                 lane_data, point_count * 2 * sizeof(float));
        }

        // Shadow (x, y) pairs on the CPU so draw_thumbnail (called from a later
        // frame phase) can visualize them without touching the GPU buffer.
        thumb_points_.assign(lane_data,
                             lane_data ? lane_data + point_count * 2 : lane_data);

        ctx->custom_outputs[0] = &mesh_;
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        uint32_t n = static_cast<uint32_t>(thumb_points_.size() / 2);
        if (n == 0) {
            vivid::thumb_instances::draw_scatter(ctx, nullptr, 0, "Cloud");
            return;
        }
        // Convert (x, y) pairs into InstanceData3D for the shared helper.
        // z = 0 (XY plane), scale = 1, color from operator's r/g/b.
        std::vector<vivid::gpu::InstanceData3D> view(n);
        float cr = r.value, cg = g.value, cb = b.value;
        for (uint32_t i = 0; i < n; ++i) {
            auto& inst = view[i];
            inst.position[0] = thumb_points_[i * 2 + 0];
            inst.position[1] = 0.0f;
            inst.position[2] = thumb_points_[i * 2 + 1];
            inst.scale[0] = inst.scale[1] = inst.scale[2] = 1.0f;
            inst.rotation_x = inst.rotation_y = 0.0f;
            inst.color[0] = cr;
            inst.color[1] = cg;
            inst.color[2] = cb;
            inst.color[3] = 1.0f;
        }
        vivid::thumb_instances::draw_scatter(ctx, view.data(), n, "Cloud");
    }

    ~PointCloud() override {
        vivid::gpu::release(vertex_buf_);
    }

private:
    WGPUBuffer           vertex_buf_   = nullptr;
    VividMesh            mesh_{};
    VividVertexAttribute attrib_{};
    uint32_t             built_count_  = 0xFFFFFFFFu;
    std::vector<float>   thumb_points_; // shadowed (x,y) pairs for draw_thumbnail

    void rebuild(const VividGpuContext* ctx, uint32_t point_count) {
        vivid::gpu::release(vertex_buf_);
        built_count_ = point_count;

        if (point_count == 0) {
            mesh_ = VividMesh{};
            return;
        }

        uint64_t vbytes = static_cast<uint64_t>(point_count) * 2 * sizeof(float);
        WGPUBufferDescriptor bd{};
        bd.label = vivid_sv("PointCloud Vertices");
        bd.size  = vbytes;
        bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
        vertex_buf_ = wgpuDeviceCreateBuffer(ctx->device, &bd);

        attrib_ = {WGPUVertexFormat_Float32x2, 0, 0};

        mesh_ = VividMesh{};
        mesh_.vertex_buffer        = vertex_buf_;
        mesh_.vertex_buffer_offset = 0;
        mesh_.vertex_count         = point_count;
        mesh_.vertex_stride        = 2 * sizeof(float);  // 8 bytes
        mesh_.index_buffer         = nullptr;
        mesh_.index_format         = WGPUIndexFormat_Undefined;
        mesh_.index_count          = 0;
        mesh_.topology             = WGPUPrimitiveTopology_PointList;
        mesh_.attributes           = &attrib_;
        mesh_.attribute_count      = 1;
    }
};

VIVID_REGISTER(PointCloud)
VIVID_THUMBNAIL(PointCloud)

VIVID_DESCRIBE_REF_TYPE(VividMesh)
