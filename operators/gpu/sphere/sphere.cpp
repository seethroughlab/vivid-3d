#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_common.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// =============================================================================
// Sphere — UV sphere mesh generator
//
// Output: "mesh" (VIVID_PORT_GPU_MESH)
// Vertices: (lat_segments+1)*(lon_segments+1), interleaved
//           pos(vec3) + normal(vec3) + uv(vec2) = 32 bytes.
// Indices:  lat_segments * lon_segments * 6 uint32_t, TriangleList.
// Lazy-rebuilds when lat_segments / lon_segments params change.
// =============================================================================

struct Sphere : vivid::GpuOperatorBase {
    static constexpr const char* kName   = "Sphere";
    static constexpr bool kTimeDependent = false;

    vivid::Param<int> lat_segments {"lat_segments", 16, 4, 128};
    vivid::Param<int> lon_segments {"lon_segments", 16, 4, 128};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        out.push_back(&lat_segments);
        out.push_back(&lon_segments);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"mesh", VIVID_PORT_GPU_MESH, VIVID_PORT_OUTPUT});
    }

    void process_gpu(const VividGpuContext* ctx) override {
        if (ctx->output_mesh_count == 0) return;

        int lat = lat_segments.int_value();
        int lon = lon_segments.int_value();

        if (lat != built_lat_ || lon != built_lon_) {
            rebuild(ctx, lat, lon);
        }

        ctx->output_meshes[0] = &mesh_;
    }

    ~Sphere() override {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
    }

private:
    WGPUBuffer vertex_buf_ = nullptr;
    WGPUBuffer index_buf_  = nullptr;
    VividMesh  mesh_{};
    VividVertexAttribute attribs_[3]{};
    int built_lat_ = -1;
    int built_lon_ = -1;

    void rebuild(const VividGpuContext* ctx, int lat, int lon) {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
        built_lat_ = lat;
        built_lon_ = lon;

        // (lat+1) latitude rings, (lon+1) vertices per ring (first == last seam)
        uint32_t vertex_count = static_cast<uint32_t>((lat + 1) * (lon + 1));
        uint32_t index_count  = static_cast<uint32_t>(lat * lon * 6);

        // 8 floats per vertex: pos(3) + normal(3) + uv(2) = 32 bytes
        std::vector<float> vdata(vertex_count * 8);
        uint32_t vi = 0;
        for (int i = 0; i <= lat; ++i) {
            float phi     = static_cast<float>(i) / static_cast<float>(lat) * 3.14159265358979f;
            float sin_phi = std::sin(phi);
            float cos_phi = std::cos(phi);
            float v_coord = static_cast<float>(i) / static_cast<float>(lat);
            for (int j = 0; j <= lon; ++j) {
                float theta     = static_cast<float>(j) / static_cast<float>(lon) * 6.28318530717959f;
                float sin_theta = std::sin(theta);
                float cos_theta = std::cos(theta);
                float u_coord   = static_cast<float>(j) / static_cast<float>(lon);

                // On a unit sphere, position == normal
                float nx = sin_phi * cos_theta;
                float ny = cos_phi;
                float nz = sin_phi * sin_theta;

                vdata[vi++] = nx;      // pos.x
                vdata[vi++] = ny;      // pos.y
                vdata[vi++] = nz;      // pos.z
                vdata[vi++] = nx;      // normal.x
                vdata[vi++] = ny;      // normal.y
                vdata[vi++] = nz;      // normal.z
                vdata[vi++] = u_coord; // uv.x
                vdata[vi++] = v_coord; // uv.y
            }
        }

        std::vector<uint32_t> idata(index_count);
        uint32_t idx = 0;
        for (int i = 0; i < lat; ++i) {
            for (int j = 0; j < lon; ++j) {
                uint32_t tl = static_cast<uint32_t>(i * (lon + 1) + j);
                uint32_t tr = tl + 1;
                uint32_t bl = tl + static_cast<uint32_t>(lon + 1);
                uint32_t br = bl + 1;
                idata[idx++] = tl; idata[idx++] = bl; idata[idx++] = br;
                idata[idx++] = tl; idata[idx++] = br; idata[idx++] = tr;
            }
        }

        uint64_t vbytes = vertex_count * 8 * sizeof(float);
        {
            WGPUBufferDescriptor bd{};
            bd.label = vivid_sv("Sphere Vertices");
            bd.size  = vbytes;
            bd.usage = WGPUBufferUsage_Vertex | WGPUBufferUsage_CopyDst;
            vertex_buf_ = wgpuDeviceCreateBuffer(ctx->device, &bd);
            wgpuQueueWriteBuffer(ctx->queue, vertex_buf_, 0, vdata.data(), vbytes);
        }

        uint64_t ibytes = index_count * sizeof(uint32_t);
        {
            WGPUBufferDescriptor bd{};
            bd.label = vivid_sv("Sphere Indices");
            bd.size  = ibytes;
            bd.usage = WGPUBufferUsage_Index | WGPUBufferUsage_CopyDst;
            index_buf_ = wgpuDeviceCreateBuffer(ctx->device, &bd);
            wgpuQueueWriteBuffer(ctx->queue, index_buf_, 0, idata.data(), ibytes);
        }

        attribs_[0] = {WGPUVertexFormat_Float32x3,  0, 0};  // pos
        attribs_[1] = {WGPUVertexFormat_Float32x3, 12, 1};  // normal
        attribs_[2] = {WGPUVertexFormat_Float32x2, 24, 2};  // uv

        mesh_ = VividMesh{};
        mesh_.vertex_buffer        = vertex_buf_;
        mesh_.vertex_buffer_offset = 0;
        mesh_.vertex_count         = vertex_count;
        mesh_.vertex_stride        = 8 * sizeof(float);  // 32 bytes
        mesh_.index_buffer         = index_buf_;
        mesh_.index_format         = WGPUIndexFormat_Uint32;
        mesh_.index_count          = index_count;
        mesh_.topology             = WGPUPrimitiveTopology_TriangleList;
        mesh_.attributes           = attribs_;
        mesh_.attribute_count      = 3;
    }
};

VIVID_REGISTER(Sphere)
