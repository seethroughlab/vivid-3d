#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/mesh_builder.h"
#include "operator_api/thumbnail_3d_gpu.h"
#include <cstdio>
#include <cmath>
#include <vector>

// =============================================================================
// Sweep — profile-along-path geometry generator
//
// Ported from legacy vivid-render3d sweep.h, using linmath.h math
// and MeshBuilderUtil for geometry construction.
// =============================================================================

static constexpr float kPi  = 3.14159265358979323846f;
static constexpr float kTau = 6.28318530717958647692f;

// ---------------------------------------------------------------------------
// Path evaluation helpers
// ---------------------------------------------------------------------------

struct Vec3 { float x, y, z; };

static Vec3 vec3_add(Vec3 a, Vec3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
static Vec3 vec3_scale(Vec3 v, float s) { return {v.x*s, v.y*s, v.z*s}; }
static Vec3 vec3_cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
static float vec3_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static float vec3_len(Vec3 v) { return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z); }
static Vec3 vec3_normalize(Vec3 v) {
    float l = vec3_len(v);
    if (l < 1e-8f) return {0,0,0};
    return {v.x/l, v.y/l, v.z/l};
}

// Path types
enum PathType { PATH_LINE=0, PATH_HELIX=1, PATH_CIRCLE=2, PATH_ARC=3 };
// Profile types
enum ProfileType { PROF_CIRCLE=0, PROF_SQUARE=1, PROF_STAR=2, PROF_TRIANGLE=3 };

struct PathParams {
    int path_type;
    float height, radius, turns;
    int path_segments;
    float arc_angle;
};

static Vec3 evaluate_path(const PathParams& p, float t) {
    switch (p.path_type) {
    case PATH_LINE: {
        float h = p.height;
        return {0.0f, t * h - h * 0.5f, 0.0f};
    }
    case PATH_HELIX: {
        float h = p.height;
        float r = p.radius;
        float angle = t * p.turns * kTau;
        return {r * std::cos(angle), t * h - h * 0.5f, r * std::sin(angle)};
    }
    case PATH_CIRCLE: {
        float r = p.radius;
        float angle = t * kTau;
        return {r * std::cos(angle), 0.0f, r * std::sin(angle)};
    }
    case PATH_ARC: {
        float r = p.radius;
        float angle = t * p.arc_angle - p.arc_angle * 0.5f;
        return {r * std::cos(angle), 0.0f, r * std::sin(angle)};
    }
    }
    return {0,0,0};
}

// Compute Frenet frame via finite differences
static void compute_frame(const PathParams& p, float t, Vec3& T, Vec3& N, Vec3& B) {
    float eps = 0.0001f;
    float t0 = std::max(0.0f, t - eps);
    float t1 = std::min(1.0f, t + eps);
    Vec3 p0 = evaluate_path(p, t0);
    Vec3 p1 = evaluate_path(p, t1);
    T = vec3_normalize({p1.x-p0.x, p1.y-p0.y, p1.z-p0.z});

    // Handle degenerate case (straight line)
    if (p.path_type == PATH_LINE) {
        T = {0.0f, 1.0f, 0.0f};
        N = {1.0f, 0.0f, 0.0f};
        B = {0.0f, 0.0f, 1.0f};
        return;
    }

    // For curved paths, use a reference vector
    Vec3 ref = {0.0f, 1.0f, 0.0f};
    if (std::abs(vec3_dot(T, ref)) > 0.99f) {
        ref = {1.0f, 0.0f, 0.0f};
    }
    B = vec3_normalize(vec3_cross(T, ref));
    N = vec3_normalize(vec3_cross(B, T));
}

// Generate 2D profile points
struct Vec2 { float x, y; };

static std::vector<Vec2> generate_profile(int profile_type, float profile_radius, int profile_segments) {
    std::vector<Vec2> pts;
    float r = profile_radius;

    switch (profile_type) {
    case PROF_CIRCLE:
        for (int i = 0; i < profile_segments; ++i) {
            float angle = static_cast<float>(i) / static_cast<float>(profile_segments) * kTau;
            pts.push_back({r * std::cos(angle), r * std::sin(angle)});
        }
        break;
    case PROF_SQUARE:
        pts.push_back({-r, -r});
        pts.push_back({ r, -r});
        pts.push_back({ r,  r});
        pts.push_back({-r,  r});
        break;
    case PROF_STAR: {
        int star_points = 5;
        float inner_r = r * 0.4f;
        for (int i = 0; i < star_points * 2; ++i) {
            float angle = static_cast<float>(i) / static_cast<float>(star_points * 2) * kTau
                          - kPi * 0.5f;
            float rad = (i % 2 == 0) ? r : inner_r;
            pts.push_back({rad * std::cos(angle), rad * std::sin(angle)});
        }
        break;
    }
    case PROF_TRIANGLE:
        for (int i = 0; i < 3; ++i) {
            float angle = static_cast<float>(i) / 3.0f * kTau - kPi * 0.5f;
            pts.push_back({r * std::cos(angle), r * std::sin(angle)});
        }
        break;
    }
    return pts;
}

static int get_profile_seg_count(int profile_type, int profile_segments) {
    switch (profile_type) {
    case PROF_CIRCLE:   return profile_segments;
    case PROF_SQUARE:   return 4;
    case PROF_STAR:     return 10;
    case PROF_TRIANGLE: return 3;
    }
    return profile_segments;
}

// =============================================================================
// Sweep Operator
// =============================================================================

struct Sweep : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "Sweep";
    static constexpr bool kTimeDependent = false;

    // Path params
    vivid::Param<int>   path_type     {"path_type",     0, {"Line", "Helix", "Circle", "Arc"}};
    vivid::Param<float> path_height   {"path_height",   2.0f,  0.01f, 100.0f};
    vivid::Param<float> path_radius   {"path_radius",   1.0f,  0.01f, 100.0f};
    vivid::Param<float> path_turns    {"path_turns",    1.0f,  0.0f,  20.0f};
    vivid::Param<int>   path_segments {"path_segments", 32, 3, 256};
    vivid::Param<float> arc_angle     {"arc_angle",     kPi,   0.01f, kTau};

    // Profile params
    vivid::Param<int>   profile_type     {"profile_type",     0, {"Circle", "Square", "Star", "Triangle"}};
    vivid::Param<float> profile_radius   {"profile_radius",   0.2f, 0.001f, 50.0f};
    vivid::Param<int>   profile_segments {"profile_segments", 16, 3, 64};

    // Modifiers
    vivid::Param<float> twist       {"twist",       0.0f, -12.566f, 12.566f};
    vivid::Param<float> scale_start {"scale_start", 1.0f, 0.01f, 5.0f};
    vivid::Param<float> scale_end   {"scale_end",   1.0f, 0.01f, 5.0f};
    vivid::Param<int>   caps        {"caps",        1, {"Off", "On"}};

    // Material
    vivid::Param<float> r         {"r",         0.8f, 0.0f, 1.0f};
    vivid::Param<float> g         {"g",         0.8f, 0.0f, 1.0f};
    vivid::Param<float> b         {"b",         0.8f, 0.0f, 1.0f};
    vivid::Param<float> a_color   {"a",         1.0f, 0.0f, 1.0f};
    vivid::Param<float> roughness {"roughness",  0.5f, 0.0f, 1.0f};
    vivid::Param<float> metallic  {"metallic",   0.0f, 0.0f, 1.0f};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        vivid::param_group(path_type, "Path");
        vivid::param_group(path_height, "Path");
        vivid::param_group(path_radius, "Path");
        vivid::param_group(path_turns, "Path");
        vivid::param_group(path_segments, "Path");
        vivid::param_group(arc_angle, "Path");

        vivid::param_group(profile_type, "Profile");
        vivid::param_group(profile_radius, "Profile");
        vivid::param_group(profile_segments, "Profile");

        vivid::param_group(twist, "Modifiers");
        vivid::param_group(scale_start, "Modifiers");
        vivid::param_group(scale_end, "Modifiers");
        vivid::param_group(caps, "Modifiers");

        vivid::param_group(r, "Color");
        vivid::param_group(g, "Color");
        vivid::param_group(b, "Color");
        vivid::param_group(a_color, "Color");
        vivid::display_hint(r, VIVID_DISPLAY_COLOR);
        vivid::display_hint(g, VIVID_DISPLAY_COLOR);
        vivid::display_hint(b, VIVID_DISPLAY_COLOR);
        vivid::display_hint(a_color, VIVID_DISPLAY_COLOR);

        vivid::param_group(roughness, "Material");
        vivid::param_group(metallic, "Material");

        out.push_back(&path_type);
        out.push_back(&path_height);
        out.push_back(&path_radius);
        out.push_back(&path_turns);
        out.push_back(&path_segments);
        out.push_back(&arc_angle);
        out.push_back(&profile_type);
        out.push_back(&profile_radius);
        out.push_back(&profile_segments);
        out.push_back(&twist);
        out.push_back(&scale_start);
        out.push_back(&scale_end);
        out.push_back(&caps);
        out.push_back(&r);
        out.push_back(&g);
        out.push_back(&b);
        out.push_back(&a_color);
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

        // Check if any param changed
        bool dirty = (built_path_type_ != path_type.int_value() ||
                      built_path_segs_ != path_segments.int_value() ||
                      built_prof_type_ != profile_type.int_value() ||
                      built_prof_segs_ != profile_segments.int_value() ||
                      built_path_h_ != path_height.value ||
                      built_path_r_ != path_radius.value ||
                      built_turns_ != path_turns.value ||
                      built_arc_ != arc_angle.value ||
                      built_prof_r_ != profile_radius.value ||
                      built_twist_ != twist.value ||
                      built_scale_s_ != scale_start.value ||
                      built_scale_e_ != scale_end.value ||
                      built_caps_ != caps.int_value());

        if (dirty) {
            rebuild(ctx);
        }

        fragment_.color[0] = r.value;
        fragment_.color[1] = g.value;
        fragment_.color[2] = b.value;
        fragment_.color[3] = a_color.value;
        fragment_.roughness = roughness.value;
        fragment_.metallic  = metallic.value;
        vivid::gpu::scene_fragment_identity(fragment_);

        fragment_.pipeline       = nullptr;
        fragment_.material_binds = nullptr;
        fragment_.children       = nullptr;
        fragment_.child_count    = 0;

        ctx->custom_outputs[0] = &fragment_;
    }

    ~Sweep() override {
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

    int built_path_type_ = -1, built_path_segs_ = -1;
    int built_prof_type_ = -1, built_prof_segs_ = -1;
    float built_path_h_ = -1, built_path_r_ = -1, built_turns_ = -1, built_arc_ = -1;
    float built_prof_r_ = -1, built_twist_ = -1, built_scale_s_ = -1, built_scale_e_ = -1;
    int built_caps_ = -1;

    void rebuild(const VividGpuContext* ctx) {
        vivid::gpu::release(vertex_buf_);
        vivid::gpu::release(index_buf_);
        vertex_buf_ = nullptr;
        index_buf_ = nullptr;

        vivid::gpu::MeshBuilderUtil mesh;

        PathParams pp{};
        pp.path_type = path_type.int_value();
        pp.height = path_height.value;
        pp.radius = path_radius.value;
        pp.turns = path_turns.value;
        pp.path_segments = path_segments.int_value();
        pp.arc_angle = arc_angle.value;

        int profType = profile_type.int_value();
        float profR = profile_radius.value;
        int profSegs = profile_segments.int_value();
        int pathSegs = pp.path_segments;

        std::vector<Vec2> profile = generate_profile(profType, profR, profSegs);
        int profCount = get_profile_seg_count(profType, profSegs);

        bool pathClosed = (pp.path_type == PATH_CIRCLE);

        // Generate vertices by sweeping profile along path
        for (int i = 0; i <= pathSegs; ++i) {
            float t = static_cast<float>(i) / static_cast<float>(pathSegs);

            // Skip last ring for closed paths
            if (pathClosed && i == pathSegs) continue;

            Vec3 pos = evaluate_path(pp, t);
            Vec3 T, N, B;
            compute_frame(pp, t, T, N, B);

            float sc = scale_start.value + (scale_end.value - scale_start.value) * t;
            float twistAngle = t * twist.value;
            float cosT = std::cos(twistAngle), sinT = std::sin(twistAngle);

            for (int j = 0; j < profCount; ++j) {
                Vec2 p = profile[j];
                p.x *= sc;
                p.y *= sc;

                // Apply twist in N-B plane
                Vec2 twisted = {p.x * cosT - p.y * sinT, p.x * sinT + p.y * cosT};

                // Transform to world space
                Vec3 worldPos = vec3_add(pos, vec3_add(vec3_scale(N, twisted.x), vec3_scale(B, twisted.y)));

                // Normal: outward from profile center
                Vec3 localNorm = vec3_normalize({twisted.x, twisted.y, 0.0f});
                Vec3 worldNorm = vec3_add(vec3_scale(N, localNorm.x), vec3_scale(B, localNorm.y));
                worldNorm = vec3_normalize(worldNorm);

                float u = static_cast<float>(j) / static_cast<float>(profCount);
                float v = t;

                mesh.addVertex(worldPos.x, worldPos.y, worldPos.z,
                               worldNorm.x, worldNorm.y, worldNorm.z,
                               u, v);
            }
        }

        // Generate faces connecting adjacent rings
        int ringCount = pathClosed ? pathSegs : pathSegs + 1;
        for (int i = 0; i < (pathClosed ? pathSegs : pathSegs); ++i) {
            int nextRing = (i + 1) % ringCount;

            for (int j = 0; j < profCount; ++j) {
                int nextProf = (j + 1) % profCount;

                uint32_t aa = static_cast<uint32_t>(i * profCount + j);
                uint32_t bb = static_cast<uint32_t>(i * profCount + nextProf);
                uint32_t cc = static_cast<uint32_t>(nextRing * profCount + nextProf);
                uint32_t dd = static_cast<uint32_t>(nextRing * profCount + j);

                mesh.addQuad(aa, bb, cc, dd);
            }
        }

        // Add caps for open paths
        bool doCaps = caps.int_value() != 0;
        if (!pathClosed && doCaps) {
            add_cap(mesh, pp, profile, profCount, true);
            add_cap(mesh, pp, profile, profCount, false);
        }

        mesh.computeNormals();
        mesh.computeTangents();

        auto bufs = mesh.uploadBuffers(ctx->device, ctx->queue, "Sweep");
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

        // Bounding box
        bmin_[0] = bmin_[1] = bmin_[2] = 1e9f;
        bmax_[0] = bmax_[1] = bmax_[2] = -1e9f;
        for (const auto& v : cpu_verts_) {
            for (int i = 0; i < 3; ++i) {
                if (v.position[i] < bmin_[i]) bmin_[i] = v.position[i];
                if (v.position[i] > bmax_[i]) bmax_[i] = v.position[i];
            }
        }

        // Cache built params
        built_path_type_ = path_type.int_value();
        built_path_segs_ = path_segments.int_value();
        built_prof_type_ = profile_type.int_value();
        built_prof_segs_ = profile_segments.int_value();
        built_path_h_ = path_height.value;
        built_path_r_ = path_radius.value;
        built_turns_ = path_turns.value;
        built_arc_ = arc_angle.value;
        built_prof_r_ = profile_radius.value;
        built_twist_ = twist.value;
        built_scale_s_ = scale_start.value;
        built_scale_e_ = scale_end.value;
        built_caps_ = caps.int_value();
    }

    void add_cap(vivid::gpu::MeshBuilderUtil& mesh, const PathParams& pp,
                 const std::vector<Vec2>& profile, int profCount, bool isStart) {
        float t = isStart ? 0.0f : 1.0f;
        Vec3 pos = evaluate_path(pp, t);
        Vec3 T, N, B;
        compute_frame(pp, t, T, N, B);

        float sc = isStart ? scale_start.value : scale_end.value;
        float twistAngle = t * twist.value;
        float cosT = std::cos(twistAngle), sinT = std::sin(twistAngle);

        Vec3 capNormal = isStart ? Vec3{-T.x, -T.y, -T.z} : T;

        // Center vertex
        uint32_t centerIdx = mesh.addVertex(pos.x, pos.y, pos.z,
                                            capNormal.x, capNormal.y, capNormal.z,
                                            0.5f, 0.5f);

        // Edge vertices
        std::vector<uint32_t> edgeIdx;
        for (int j = 0; j < profCount; ++j) {
            Vec2 p = profile[j];
            p.x *= sc;
            p.y *= sc;
            Vec2 twisted = {p.x * cosT - p.y * sinT, p.x * sinT + p.y * cosT};
            Vec3 wp = vec3_add(pos, vec3_add(vec3_scale(N, twisted.x), vec3_scale(B, twisted.y)));

            float u = 0.5f + twisted.x / (profile_radius.value * 2.0f);
            float v = 0.5f + twisted.y / (profile_radius.value * 2.0f);

            edgeIdx.push_back(mesh.addVertex(wp.x, wp.y, wp.z,
                                             capNormal.x, capNormal.y, capNormal.z,
                                             u, v));
        }

        for (int j = 0; j < profCount; ++j) {
            int nextJ = (j + 1) % profCount;
            if (isStart) {
                mesh.addTriangle(centerIdx, edgeIdx[nextJ], edgeIdx[j]);
            } else {
                mesh.addTriangle(centerIdx, edgeIdx[j], edgeIdx[nextJ]);
            }
        }
    }
};

VIVID_REGISTER(Sweep)
VIVID_THUMBNAIL(Sweep)

VIVID_DESCRIBE_REF_TYPE(vivid::gpu::VividSceneFragment)
