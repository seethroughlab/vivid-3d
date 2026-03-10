#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_common.h"
#include <cstdio>
#include <cstring>
#include <vector>

// =============================================================================
// Box — axis-aligned box mesh generator with per-face normals
//
// Output: "mesh" (VIVID_PORT_GPU_MESH)
// 24 vertices (4 per face × 6 faces), same layout as Sphere:
//   pos(vec3) + normal(vec3) + uv(vec2) = 32 bytes.
// 36 indices (6 per face), TriangleList.
// Lazy-rebuilds when width/height/depth params change.
// =============================================================================

struct Box : vivid::GpuOperatorBase {
    static constexpr const char* kName   = "Box";
    static constexpr bool kTimeDependent = false;

    vivid::Param<float> width  {"width",  1.0f, 0.01f, 10.0f};
    vivid::Param<float> height {"height", 1.0f, 0.01f, 10.0f};
    vivid::Param<float> depth  {"depth",  1.0f, 0.01f, 10.0f};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        out.push_back(&width);
        out.push_back(&height);
        out.push_back(&depth);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"mesh", VIVID_PORT_GPU_MESH, VIVID_PORT_OUTPUT});
    }

    void process_gpu(const VividGpuContext* ctx) override {
        if (ctx->output_mesh_count == 0) return;

        float w = width.value, h = height.value, d = depth.value;
        if (w != built_w_ || h != built_h_ || d != built_d_) {
            rebuild(ctx, w, h, d);
        }

        ctx->output_meshes[0] = &mesh_;
    }

    ~Box() override {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
    }

private:
    WGPUBuffer vertex_buf_ = nullptr;
    WGPUBuffer index_buf_  = nullptr;
    VividMesh  mesh_{};
    VividVertexAttribute attribs_[3]{};
    float built_w_ = -1.0f, built_h_ = -1.0f, built_d_ = -1.0f;

    void rebuild(const VividGpuContext* ctx, float w, float h, float d) {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
        built_w_ = w; built_h_ = h; built_d_ = d;

        float hw = w * 0.5f, hh = h * 0.5f, hd = d * 0.5f;

        // 6 faces × 4 vertices × 8 floats (pos + normal + uv)
        struct V { float px, py, pz, nx, ny, nz, u, v; };
        V verts[24] = {
            // +X face (right)
            { hw, -hh,  hd,  1, 0, 0,  0, 1},
            { hw,  hh,  hd,  1, 0, 0,  0, 0},
            { hw,  hh, -hd,  1, 0, 0,  1, 0},
            { hw, -hh, -hd,  1, 0, 0,  1, 1},
            // -X face (left)
            {-hw, -hh, -hd, -1, 0, 0,  0, 1},
            {-hw,  hh, -hd, -1, 0, 0,  0, 0},
            {-hw,  hh,  hd, -1, 0, 0,  1, 0},
            {-hw, -hh,  hd, -1, 0, 0,  1, 1},
            // +Y face (top)
            {-hw,  hh,  hd,  0, 1, 0,  0, 1},
            { hw,  hh,  hd,  0, 1, 0,  1, 1},
            { hw,  hh, -hd,  0, 1, 0,  1, 0},
            {-hw,  hh, -hd,  0, 1, 0,  0, 0},
            // -Y face (bottom)
            {-hw, -hh, -hd,  0,-1, 0,  0, 1},
            { hw, -hh, -hd,  0,-1, 0,  1, 1},
            { hw, -hh,  hd,  0,-1, 0,  1, 0},
            {-hw, -hh,  hd,  0,-1, 0,  0, 0},
            // +Z face (front)
            {-hw, -hh,  hd,  0, 0, 1,  0, 1},
            {-hw,  hh,  hd,  0, 0, 1,  0, 0},
            { hw,  hh,  hd,  0, 0, 1,  1, 0},
            { hw, -hh,  hd,  0, 0, 1,  1, 1},
            // -Z face (back)
            { hw, -hh, -hd,  0, 0,-1,  0, 1},
            { hw,  hh, -hd,  0, 0,-1,  0, 0},
            {-hw,  hh, -hd,  0, 0,-1,  1, 0},
            {-hw, -hh, -hd,  0, 0,-1,  1, 1},
        };

        // 6 faces × 6 indices (two triangles per face, CCW)
        uint32_t idata[36];
        for (uint32_t f = 0; f < 6; ++f) {
            uint32_t b = f * 4;
            uint32_t o = f * 6;
            idata[o+0] = b+0; idata[o+1] = b+1; idata[o+2] = b+2;
            idata[o+3] = b+0; idata[o+4] = b+2; idata[o+5] = b+3;
        }

        uint64_t vbytes = sizeof(verts);
        {
            WGPUBufferDescriptor bd{};
            bd.label = vivid_sv("Box Vertices");
            bd.size  = vbytes;
            bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            vertex_buf_ = wgpuDeviceCreateBuffer(ctx->device, &bd);
            wgpuQueueWriteBuffer(ctx->queue, vertex_buf_, 0, verts, vbytes);
        }

        uint64_t ibytes = sizeof(idata);
        {
            WGPUBufferDescriptor bd{};
            bd.label = vivid_sv("Box Indices");
            bd.size  = ibytes;
            bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
            index_buf_ = wgpuDeviceCreateBuffer(ctx->device, &bd);
            wgpuQueueWriteBuffer(ctx->queue, index_buf_, 0, idata, ibytes);
        }

        attribs_[0] = {WGPUVertexFormat_Float32x3,  0, 0};  // pos
        attribs_[1] = {WGPUVertexFormat_Float32x3, 12, 1};  // normal
        attribs_[2] = {WGPUVertexFormat_Float32x2, 24, 2};  // uv

        mesh_ = VividMesh{};
        mesh_.vertex_buffer        = vertex_buf_;
        mesh_.vertex_buffer_offset = 0;
        mesh_.vertex_count         = 24;
        mesh_.vertex_stride        = 8 * sizeof(float);  // 32 bytes
        mesh_.index_buffer         = index_buf_;
        mesh_.index_format         = WGPUIndexFormat_Uint32;
        mesh_.index_count          = 36;
        mesh_.topology             = WGPUPrimitiveTopology_TriangleList;
        mesh_.attributes           = attribs_;
        mesh_.attribute_count      = 3;
    }
};

VIVID_REGISTER(Box)
