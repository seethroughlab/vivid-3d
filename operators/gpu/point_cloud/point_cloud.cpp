#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_common.h"
#include <cstdio>
#include <cstring>
#include <vector>

// =============================================================================
// PointCloud — interprets a CONTROL_SPREAD as [x0,y0, x1,y1, ...] pairs
//              and produces a PointList mesh.
//
// Input:  "positions" (VIVID_PORT_CONTROL_SPREAD)
// Output: "mesh"      (VIVID_PORT_GPU_MESH, topology PointList)
//
// Vertex layout: vec2f (xy) = 8 bytes per point.
// Rebuilds vertex buffer when point count changes; uploads spread data each tick.
// =============================================================================

struct PointCloud : vivid::GpuOperatorBase {
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
        out.push_back({"positions", VIVID_PORT_CONTROL_SPREAD, VIVID_PORT_INPUT});
        out.push_back({"mesh",      VIVID_PORT_GPU_MESH,       VIVID_PORT_OUTPUT});
    }

    void process_gpu(const VividGpuContext* ctx) override {
        if (ctx->output_mesh_count == 0) return;

        // Read positions spread: pairs of [x,y]
        uint32_t spread_len = 0;
        const float* spread_data = nullptr;
        if (ctx->input_spreads && ctx->input_spreads[0].length > 0) {
            spread_len  = ctx->input_spreads[0].length;
            spread_data = ctx->input_spreads[0].data;
        }

        uint32_t point_count = spread_len / 2;

        if (point_count != built_count_) {
            rebuild(ctx, point_count);
        }

        // Upload current spread data each tick
        if (vertex_buf_ && spread_data && point_count > 0) {
            wgpuQueueWriteBuffer(ctx->queue, vertex_buf_, 0,
                                 spread_data, point_count * 2 * sizeof(float));
        }

        ctx->output_meshes[0] = &mesh_;
    }

    ~PointCloud() override {
        vivid::gpu::release(vertex_buf_);
    }

private:
    WGPUBuffer           vertex_buf_   = nullptr;
    VividMesh            mesh_{};
    VividVertexAttribute attrib_{};
    uint32_t             built_count_  = 0xFFFFFFFFu;

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
