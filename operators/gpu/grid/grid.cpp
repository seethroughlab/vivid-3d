#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_common.h"
#include "operator_api/type_id.h"
#include "operator_api/port_type_registry.h"
#include "operator_api/thumbnail_3d.h"
#include <cstdio>
#include <cstring>
#include <vector>

// =============================================================================
// Grid — subdivided plane mesh generator
//
// Output: "mesh" (VIVID_CUSTOM_PORT)
// Vertices: (cols+1)*(rows+1), interleaved pos(vec3) + uv(vec2) = 20 bytes.
// Indices:  cols*rows*6 uint32_t, TriangleList.
// Lazy-rebuilds buffers when cols/rows params change.
// =============================================================================

struct Grid : vivid::GpuOperatorBase {
    static constexpr const char* kName   = "Grid";
    static constexpr bool kTimeDependent = false;

    vivid::Param<int> cols {"cols", 8, 1, 256};
    vivid::Param<int> rows {"rows", 8, 1, 256};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        out.push_back(&cols);
        out.push_back(&rows);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(VIVID_CUSTOM_REF_PORT("mesh", VIVID_PORT_OUTPUT, VividMesh));
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        if (!ctx || !ctx->pixels || cpu_verts_.empty()) return;
        float bmin[3] = {-1, -1, 0}, bmax[3] = {1, 1, 0}; // flat XY plane
        auto cam = vivid::thumb3d::camera_from_bounds(bmin, bmax,
                                                       ctx->width, ctx->height);
        vivid::thumb3d::render_mesh(ctx->pixels, ctx->width, ctx->height, ctx->stride,
            cpu_verts_.data(), static_cast<uint32_t>(cpu_verts_.size() / 5),
            cpu_indices_.data(), static_cast<uint32_t>(cpu_indices_.size()),
            5 * sizeof(float), 0, UINT32_MAX, cam);
    }

    void process_gpu(const VividGpuContext* ctx) override {
        if (ctx->custom_output_count == 0) return;

        int c = cols.int_value();
        int r = rows.int_value();

        if (c != built_cols_ || r != built_rows_) {
            rebuild(ctx, c, r);
        }

        ctx->custom_outputs[0] = &mesh_;
    }

    ~Grid() override {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
    }

private:
    WGPUBuffer vertex_buf_ = nullptr;
    WGPUBuffer index_buf_  = nullptr;
    VividMesh  mesh_{};
    VividVertexAttribute attribs_[2]{};
    int built_cols_ = -1;
    int built_rows_ = -1;
    std::vector<float> cpu_verts_;
    std::vector<uint32_t> cpu_indices_;

    void rebuild(const VividGpuContext* ctx, int c, int r) {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
        built_cols_ = c;
        built_rows_ = r;

        uint32_t vcols = static_cast<uint32_t>(c + 1);
        uint32_t vrows = static_cast<uint32_t>(r + 1);
        uint32_t vertex_count = vcols * vrows;
        uint32_t index_count  = static_cast<uint32_t>(c * r * 6);

        // Vertex data: pos(vec3) + uv(vec2) = 5 floats = 20 bytes per vertex
        std::vector<float> vdata(vertex_count * 5);
        for (uint32_t row = 0; row < vrows; ++row) {
            float v = static_cast<float>(row) / static_cast<float>(r);
            for (uint32_t col = 0; col < vcols; ++col) {
                float u = static_cast<float>(col) / static_cast<float>(c);
                uint32_t i = (row * vcols + col) * 5;
                vdata[i + 0] = u * 2.0f - 1.0f;  // x: -1..1
                vdata[i + 1] = v * 2.0f - 1.0f;  // y: -1..1
                vdata[i + 2] = 0.0f;               // z
                vdata[i + 3] = u;                  // uv.x
                vdata[i + 4] = v;                  // uv.y
            }
        }

        // Index data: two triangles per quad, CCW winding
        std::vector<uint32_t> idata(index_count);
        uint32_t idx = 0;
        for (uint32_t row = 0; row < static_cast<uint32_t>(r); ++row) {
            for (uint32_t col = 0; col < static_cast<uint32_t>(c); ++col) {
                uint32_t tl = row * vcols + col;
                uint32_t tr = tl + 1;
                uint32_t bl = tl + vcols;
                uint32_t br = bl + 1;
                idata[idx++] = tl; idata[idx++] = bl; idata[idx++] = br;
                idata[idx++] = tl; idata[idx++] = br; idata[idx++] = tr;
            }
        }

        cpu_verts_ = vdata;
        cpu_indices_ = idata;

        uint64_t vbytes = vertex_count * 5 * sizeof(float);
        {
            WGPUBufferDescriptor bd{};
            bd.label = vivid_sv("Grid Vertices");
            bd.size  = vbytes;
            bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            vertex_buf_ = wgpuDeviceCreateBuffer(ctx->device, &bd);
            wgpuQueueWriteBuffer(ctx->queue, vertex_buf_, 0, vdata.data(), vbytes);
        }

        uint64_t ibytes = index_count * sizeof(uint32_t);
        {
            WGPUBufferDescriptor bd{};
            bd.label = vivid_sv("Grid Indices");
            bd.size  = ibytes;
            bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
            index_buf_ = wgpuDeviceCreateBuffer(ctx->device, &bd);
            wgpuQueueWriteBuffer(ctx->queue, index_buf_, 0, idata.data(), ibytes);
        }

        attribs_[0] = {WGPUVertexFormat_Float32x3,  0, 0};  // pos
        attribs_[1] = {WGPUVertexFormat_Float32x2, 12, 1};  // uv

        mesh_ = VividMesh{};
        mesh_.vertex_buffer        = vertex_buf_;
        mesh_.vertex_buffer_offset = 0;
        mesh_.vertex_count         = vertex_count;
        mesh_.vertex_stride        = 5 * sizeof(float);  // 20 bytes
        mesh_.index_buffer         = index_buf_;
        mesh_.index_format         = WGPUIndexFormat_Uint32;
        mesh_.index_count          = index_count;
        mesh_.topology             = WGPUPrimitiveTopology_TriangleList;
        mesh_.attributes           = attribs_;
        mesh_.attribute_count      = 2;
    }
};

VIVID_REGISTER(Grid)
VIVID_THUMBNAIL(Grid)

VIVID_DESCRIBE_REF_TYPE(VividMesh)
