#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/mesh_builder.h"
#include "operator_api/thumbnail_3d_gpu.h"
#include <cstdio>
#include <cmath>
#include <vector>

// =============================================================================
// MeshBuilder — parametric primitive generator (scene fragment output)
// =============================================================================

/**
 * @brief Builds common procedural meshes with integrated material settings.
 *
 * MeshBuilder is a compact geometry source for quickly generating boxes, spheres, cylinders,
 * cones, toruses, and planes without leaving the graph.
 *
 * @param primitive Primitive type to generate.
 * @param size_x Size along the X axis.
 * @param segments Tessellation level for supported primitives.
 * @param inner_radius Inner radius used by torus-like shapes.
 * @param roughness Surface roughness for the generated mesh.
 */
struct MeshBuilder : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "MeshBuilder";
    static constexpr bool kTimeDependent = false;

    vivid::Param<int>   primitive    {"primitive",    0, {"Box", "Sphere", "Cylinder", "Cone", "Torus", "Plane"}};
    vivid::Param<float> size_x      {"size_x",       1.0f, 0.01f, 50.0f};
    vivid::Param<float> size_y      {"size_y",       1.0f, 0.01f, 50.0f};
    vivid::Param<float> size_z      {"size_z",       1.0f, 0.01f, 50.0f};
    vivid::Param<int>   segments    {"segments",    16, 3, 128};
    vivid::Param<float> inner_radius {"inner_radius", 0.3f, 0.01f, 10.0f};

    // Material
    vivid::Param<float> r         {"r",         0.8f, 0.0f, 1.0f};
    vivid::Param<float> g         {"g",         0.8f, 0.0f, 1.0f};
    vivid::Param<float> b         {"b",         0.8f, 0.0f, 1.0f};
    vivid::Param<float> a         {"a",         1.0f, 0.0f, 1.0f};
    vivid::Param<float> roughness {"roughness",  0.5f, 0.0f, 1.0f};
    vivid::Param<float> metallic  {"metallic",   0.0f, 0.0f, 1.0f};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        vivid::param_group(primitive, "Shape");
        vivid::param_group(size_x, "Shape");
        vivid::param_group(size_y, "Shape");
        vivid::param_group(size_z, "Shape");
        vivid::param_group(segments, "Shape");
        vivid::param_group(inner_radius, "Shape");

        vivid::param_group(r, "Color");
        vivid::param_group(g, "Color");
        vivid::param_group(b, "Color");
        vivid::param_group(a, "Color");
        vivid::display_hint(r, VIVID_DISPLAY_COLOR);
        vivid::display_hint(g, VIVID_DISPLAY_COLOR);
        vivid::display_hint(b, VIVID_DISPLAY_COLOR);
        vivid::display_hint(a, VIVID_DISPLAY_COLOR);

        vivid::param_group(roughness, "Material");
        vivid::param_group(metallic, "Material");

        out.push_back(&primitive);
        out.push_back(&size_x);
        out.push_back(&size_y);
        out.push_back(&size_z);
        out.push_back(&segments);
        out.push_back(&inner_radius);
        out.push_back(&r);
        out.push_back(&g);
        out.push_back(&b);
        out.push_back(&a);
        out.push_back(&roughness);
        out.push_back(&metallic);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_OUTPUT));
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        if (!ctx || !fragment_.vertex_buffer || fragment_.index_count == 0) return;
        vivid::thumb3d_gpu::render_mesh(
            ctx,
            fragment_.vertex_buffer,
            fragment_.vertex_buf_size,
            fragment_.index_buffer,
            fragment_.index_count,
            sizeof(vivid::gpu::Vertex3D),
            WGPUPrimitiveTopology_TriangleList,
            bmin_, bmax_);
    }

    void process_gpu(const VividGpuContext* ctx) override {
        if (ctx->custom_output_count == 0) return;

        int prim = primitive.int_value();
        int segs = segments.int_value();
        float sx = size_x.value, sy = size_y.value, sz = size_z.value;
        float ir = inner_radius.value;

        if (prim != built_prim_ || segs != built_segs_ ||
            sx != built_sx_ || sy != built_sy_ || sz != built_sz_ || ir != built_ir_) {
            rebuild(ctx, prim, segs, sx, sy, sz, ir);
        }

        fragment_.color[0] = r.value;
        fragment_.color[1] = g.value;
        fragment_.color[2] = b.value;
        fragment_.color[3] = a.value;
        fragment_.roughness = roughness.value;
        fragment_.metallic  = metallic.value;
        vivid::gpu::scene_fragment_identity(fragment_);

        fragment_.pipeline       = nullptr;
        fragment_.material_binds = nullptr;
        fragment_.children       = nullptr;
        fragment_.child_count    = 0;

        ctx->custom_outputs[0] = &fragment_;
    }

    ~MeshBuilder() override {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
    }

private:
    WGPUBuffer vertex_buf_ = nullptr;
    WGPUBuffer index_buf_  = nullptr;
    vivid::gpu::VividSceneFragment fragment_{};
    std::vector<vivid::gpu::Vertex3D> cpu_verts_;
    std::vector<uint32_t> cpu_indices_;
    float bmin_[3] = {0,0,0};
    float bmax_[3] = {0,0,0};

    int built_prim_ = -1;
    int built_segs_ = -1;
    float built_sx_ = -1, built_sy_ = -1, built_sz_ = -1, built_ir_ = -1;

    void rebuild(const VividGpuContext* ctx, int prim, int segs,
                 float sx, float sy, float sz, float ir) {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
        vertex_buf_ = nullptr;
        index_buf_ = nullptr;

        vivid::gpu::MeshBuilderUtil mesh;
        switch (prim) {
            case 0: mesh = vivid::gpu::MeshBuilderUtil::box(sx, sy, sz); break;
            case 1: mesh = vivid::gpu::MeshBuilderUtil::sphere(sx * 0.5f, segs); break;
            case 2: mesh = vivid::gpu::MeshBuilderUtil::cylinder(sx * 0.5f, sy, segs); break;
            case 3: mesh = vivid::gpu::MeshBuilderUtil::cone(sx * 0.5f, sy, segs); break;
            case 4: mesh = vivid::gpu::MeshBuilderUtil::torus(sx * 0.5f, ir, segs, segs); break;
            case 5: mesh = vivid::gpu::MeshBuilderUtil::plane(sx, sz, segs, segs); break;
            default: mesh = vivid::gpu::MeshBuilderUtil::box(sx, sy, sz); break;
        }

        mesh.computeTangents();

        auto bufs = mesh.uploadBuffers(ctx->device, ctx->queue, "MeshBuilder");
        vertex_buf_ = bufs.vertex_buffer;
        index_buf_  = bufs.index_buffer;

        fragment_.vertex_buffer   = vertex_buf_;
        fragment_.vertex_buf_size = bufs.vertex_buf_size;
        fragment_.index_buffer    = index_buf_;
        fragment_.index_count     = bufs.index_count;

        cpu_verts_   = mesh.vertices();
        cpu_indices_ = mesh.indices();
        fragment_.cpu_vertices     = cpu_verts_.data();
        fragment_.cpu_vertex_count = static_cast<uint32_t>(cpu_verts_.size());
        fragment_.cpu_indices      = cpu_indices_.data();
        fragment_.cpu_index_count  = static_cast<uint32_t>(cpu_indices_.size());

        // Compute bounding box for thumbnail
        bmin_[0] = bmin_[1] = bmin_[2] = 1e9f;
        bmax_[0] = bmax_[1] = bmax_[2] = -1e9f;
        for (const auto& v : cpu_verts_) {
            for (int i = 0; i < 3; ++i) {
                if (v.position[i] < bmin_[i]) bmin_[i] = v.position[i];
                if (v.position[i] > bmax_[i]) bmax_[i] = v.position[i];
            }
        }

        built_prim_ = prim;
        built_segs_ = segs;
        built_sx_ = sx; built_sy_ = sy; built_sz_ = sz;
        built_ir_ = ir;
    }
};

VIVID_REGISTER(MeshBuilder)
VIVID_THUMBNAIL(MeshBuilder)

VIVID_DESCRIBE_REF_TYPE(vivid::gpu::VividSceneFragment)
