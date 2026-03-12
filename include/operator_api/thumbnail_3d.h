#pragma once

// CPU software rasterizer for 3D mesh thumbnails.
// Renders shaded triangle meshes into the VividThumbnailContext pixel buffer.
// Header-only, no dependencies beyond standard C math and operator_api/types.h.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace vivid::thumb3d {

// ---------------------------------------------------------------------------
// Camera — auto-framed from mesh AABB
// ---------------------------------------------------------------------------

struct ThumbCamera {
    float eye[3];
    float target[3];
    float up[3];
    float fov_y;    // radians
    float aspect;
    float near_z;
    float far_z;
};

inline ThumbCamera camera_from_bounds(const float bmin[3], const float bmax[3],
                                       uint32_t w, uint32_t h) {
    ThumbCamera cam{};
    float cx = (bmin[0] + bmax[0]) * 0.5f;
    float cy = (bmin[1] + bmax[1]) * 0.5f;
    float cz = (bmin[2] + bmax[2]) * 0.5f;
    float dx = bmax[0] - bmin[0];
    float dy = bmax[1] - bmin[1];
    float dz = bmax[2] - bmin[2];
    float radius = std::sqrt(dx * dx + dy * dy + dz * dz) * 0.5f;
    if (radius < 1e-6f) radius = 1.0f;

    // Position camera at upper-right-front to show 3D perspective
    float dist = radius * 2.5f;
    cam.eye[0] = cx + dist * 0.6f;
    cam.eye[1] = cy + dist * 0.4f;
    cam.eye[2] = cz + dist * 0.7f;
    cam.target[0] = cx;
    cam.target[1] = cy;
    cam.target[2] = cz;
    cam.up[0] = 0.0f;
    cam.up[1] = 1.0f;
    cam.up[2] = 0.0f;
    cam.fov_y = 0.6f; // ~34 degrees — tight framing
    cam.aspect = static_cast<float>(w) / static_cast<float>(h);
    cam.near_z = dist * 0.01f;
    cam.far_z  = dist * 4.0f;
    return cam;
}

// ---------------------------------------------------------------------------
// Light
// ---------------------------------------------------------------------------

struct ThumbLight {
    float dir[3];       // normalized, points toward light
    float color[3];     // RGB, 0-1
    float ambient[3];   // RGB ambient
};

inline ThumbLight default_light() {
    ThumbLight l{};
    // Upper-right-front key light
    float dx = 0.5f, dy = 0.7f, dz = 0.5f;
    float len = std::sqrt(dx * dx + dy * dy + dz * dz);
    l.dir[0] = dx / len; l.dir[1] = dy / len; l.dir[2] = dz / len;
    l.color[0] = 1.0f;  l.color[1] = 0.98f; l.color[2] = 0.95f;
    l.ambient[0] = 0.12f; l.ambient[1] = 0.13f; l.ambient[2] = 0.16f;
    return l;
}

// ---------------------------------------------------------------------------
// Math helpers (self-contained)
// ---------------------------------------------------------------------------

namespace detail {

struct Vec3 { float x, y, z; };
struct Vec4 { float x, y, z, w; };
struct Mat4 { float m[4][4]; };

inline Vec3 v3sub(Vec3 a, Vec3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
inline Vec3 v3cross(Vec3 a, Vec3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
inline float v3dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
inline Vec3 v3norm(Vec3 v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len < 1e-8f) return {0,1,0};
    float inv = 1.0f / len;
    return {v.x*inv, v.y*inv, v.z*inv};
}

inline Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
    Vec3 f = v3norm(v3sub(target, eye));
    Vec3 s = v3norm(v3cross(f, up));
    Vec3 u = v3cross(s, f);
    Mat4 m{};
    m.m[0][0] =  s.x; m.m[1][0] =  s.y; m.m[2][0] =  s.z; m.m[3][0] = -v3dot(s, eye);
    m.m[0][1] =  u.x; m.m[1][1] =  u.y; m.m[2][1] =  u.z; m.m[3][1] = -v3dot(u, eye);
    m.m[0][2] = -f.x; m.m[1][2] = -f.y; m.m[2][2] = -f.z; m.m[3][2] =  v3dot(f, eye);
    m.m[0][3] = 0; m.m[1][3] = 0; m.m[2][3] = 0; m.m[3][3] = 1;
    return m;
}

inline Mat4 perspective(float fov_y, float aspect, float near, float far) {
    float t = std::tan(fov_y * 0.5f);
    Mat4 m{};
    m.m[0][0] = 1.0f / (aspect * t);
    m.m[1][1] = 1.0f / t;
    m.m[2][2] = far / (near - far);
    m.m[2][3] = -1.0f;
    m.m[3][2] = (near * far) / (near - far);
    return m;
}

inline Mat4 mat4_mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k)
                r.m[i][j] += a.m[k][j] * b.m[i][k];
    return r;
}

inline Vec4 mat4_mul_vec4(const Mat4& m, Vec4 v) {
    return {
        m.m[0][0]*v.x + m.m[1][0]*v.y + m.m[2][0]*v.z + m.m[3][0]*v.w,
        m.m[0][1]*v.x + m.m[1][1]*v.y + m.m[2][1]*v.z + m.m[3][1]*v.w,
        m.m[0][2]*v.x + m.m[1][2]*v.y + m.m[2][2]*v.z + m.m[3][2]*v.w,
        m.m[0][3]*v.x + m.m[1][3]*v.y + m.m[2][3]*v.z + m.m[3][3]*v.w,
    };
}

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

inline float edge(float ax, float ay, float bx, float by, float cx, float cy) {
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

} // namespace detail

// ---------------------------------------------------------------------------
// render_mesh — the CPU rasterizer
//
// Parameters:
//   pixels          — RGBA8 output buffer (row-major)
//   w, h, stride    — pixel buffer dimensions (stride in bytes)
//   vert_data       — pointer to float vertex data
//   vert_count      — number of vertices
//   indices         — triangle index array (3 per triangle)
//   index_count     — total indices (must be multiple of 3)
//   vert_stride     — bytes between consecutive vertices
//   pos_offset      — byte offset to position (float[3]) within vertex
//   normal_offset   — byte offset to normal (float[3]) within vertex
//                     set to UINT32_MAX for no normals (uses face normal)
//   camera          — camera parameters
//   light           — light parameters (defaults to default_light())
//   color           — RGB material color, 0-1 (default: cool grey-blue)
// ---------------------------------------------------------------------------

inline void render_mesh(uint8_t* pixels, uint32_t w, uint32_t h, uint32_t stride,
                         const float* vert_data, uint32_t vert_count,
                         const uint32_t* indices, uint32_t index_count,
                         uint32_t vert_stride, uint32_t pos_offset, uint32_t normal_offset,
                         const ThumbCamera& camera,
                         const ThumbLight& light = default_light(),
                         const float color[3] = nullptr) {
    using namespace detail;

    float mat_color[3] = {0.65f, 0.72f, 0.80f}; // cool grey-blue
    if (color) { mat_color[0] = color[0]; mat_color[1] = color[1]; mat_color[2] = color[2]; }

    bool has_normals = (normal_offset != UINT32_MAX);

    // Clear to dark background
    for (uint32_t y = 0; y < h; ++y) {
        uint8_t* row = pixels + y * stride;
        for (uint32_t x = 0; x < w; ++x) {
            row[x * 4 + 0] = 18;
            row[x * 4 + 1] = 20;
            row[x * 4 + 2] = 23;
            row[x * 4 + 3] = 230;
        }
    }

    if (!vert_data || vert_count == 0 || !indices || index_count < 3) return;

    // Build view-projection matrix
    Vec3 eye = {camera.eye[0], camera.eye[1], camera.eye[2]};
    Vec3 tgt = {camera.target[0], camera.target[1], camera.target[2]};
    Vec3 up  = {camera.up[0], camera.up[1], camera.up[2]};
    Mat4 view = look_at(eye, tgt, up);
    Mat4 proj = perspective(camera.fov_y, camera.aspect, camera.near_z, camera.far_z);
    Mat4 vp   = mat4_mul(proj, view);

    // Transform vertices to clip space
    uint32_t stride_floats = vert_stride / sizeof(float);
    uint32_t pos_off_floats = pos_offset / sizeof(float);
    uint32_t norm_off_floats = normal_offset / sizeof(float);

    struct ScreenVert {
        float sx, sy, z;   // screen x, y and clip-space depth
        float nx, ny, nz;  // world-space normal
    };

    // Use heap allocation for vertex data
    auto* sv = new ScreenVert[vert_count];

    float hw = static_cast<float>(w) * 0.5f;
    float hh = static_cast<float>(h) * 0.5f;

    for (uint32_t i = 0; i < vert_count; ++i) {
        const float* vp_data = vert_data + i * stride_floats;
        const float* pos = vp_data + pos_off_floats;

        Vec4 clip = mat4_mul_vec4(vp, {pos[0], pos[1], pos[2], 1.0f});
        float inv_w = (std::fabs(clip.w) > 1e-8f) ? (1.0f / clip.w) : 0.0f;
        sv[i].sx = (clip.x * inv_w * 0.5f + 0.5f) * static_cast<float>(w);
        sv[i].sy = (1.0f - (clip.y * inv_w * 0.5f + 0.5f)) * static_cast<float>(h);
        sv[i].z  = clip.z * inv_w; // 0..1 depth

        if (has_normals) {
            const float* n = vp_data + norm_off_floats;
            sv[i].nx = n[0]; sv[i].ny = n[1]; sv[i].nz = n[2];
        }
    }

    // Z-buffer (heap allocated)
    auto* zbuf = new float[w * h];
    for (uint32_t i = 0; i < w * h; ++i) zbuf[i] = 1.0f;

    // View direction for specular
    Vec3 view_dir = v3norm(v3sub(eye, tgt));

    // Rasterize triangles
    uint32_t tri_count = index_count / 3;
    for (uint32_t t = 0; t < tri_count; ++t) {
        uint32_t i0 = indices[t * 3 + 0];
        uint32_t i1 = indices[t * 3 + 1];
        uint32_t i2 = indices[t * 3 + 2];
        if (i0 >= vert_count || i1 >= vert_count || i2 >= vert_count) continue;

        const auto& v0 = sv[i0];
        const auto& v1 = sv[i1];
        const auto& v2 = sv[i2];

        // Backface cull (screen-space, CW winding after Y flip = back-facing)
        float area = edge(v0.sx, v0.sy, v1.sx, v1.sy, v2.sx, v2.sy);
        if (area <= 0.0f) continue;

        // Clip-space depth cull
        if (v0.z < 0 && v1.z < 0 && v2.z < 0) continue;
        if (v0.z > 1 && v1.z > 1 && v2.z > 1) continue;

        // Compute face normal for meshes without vertex normals
        Vec3 fn = {0, 1, 0};
        if (!has_normals) {
            const float* p0 = vert_data + i0 * stride_floats + pos_off_floats;
            const float* p1 = vert_data + i1 * stride_floats + pos_off_floats;
            const float* p2 = vert_data + i2 * stride_floats + pos_off_floats;
            Vec3 e1 = {p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]};
            Vec3 e2 = {p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]};
            fn = v3norm(v3cross(e1, e2));
        }

        // Screen-space AABB
        float fminx = std::min({v0.sx, v1.sx, v2.sx});
        float fmaxx = std::max({v0.sx, v1.sx, v2.sx});
        float fminy = std::min({v0.sy, v1.sy, v2.sy});
        float fmaxy = std::max({v0.sy, v1.sy, v2.sy});

        int minx = std::max(0, static_cast<int>(std::floor(fminx)));
        int maxx = std::min(static_cast<int>(w) - 1, static_cast<int>(std::ceil(fmaxx)));
        int miny = std::max(0, static_cast<int>(std::floor(fminy)));
        int maxy = std::min(static_cast<int>(h) - 1, static_cast<int>(std::ceil(fmaxy)));

        float inv_area = 1.0f / area;

        for (int y = miny; y <= maxy; ++y) {
            for (int x = minx; x <= maxx; ++x) {
                float px = static_cast<float>(x) + 0.5f;
                float py = static_cast<float>(y) + 0.5f;

                float w0 = edge(v1.sx, v1.sy, v2.sx, v2.sy, px, py);
                float w1 = edge(v2.sx, v2.sy, v0.sx, v0.sy, px, py);
                float w2 = edge(v0.sx, v0.sy, v1.sx, v1.sy, px, py);

                if (w0 < 0 || w1 < 0 || w2 < 0) continue;

                w0 *= inv_area;
                w1 *= inv_area;
                w2 *= inv_area;

                // Interpolate depth
                float z = w0 * v0.z + w1 * v1.z + w2 * v2.z;
                uint32_t zi = static_cast<uint32_t>(y) * w + static_cast<uint32_t>(x);
                if (z >= zbuf[zi]) continue;
                zbuf[zi] = z;

                // Interpolate normal
                Vec3 n;
                if (has_normals) {
                    n.x = w0 * v0.nx + w1 * v1.nx + w2 * v2.nx;
                    n.y = w0 * v0.ny + w1 * v1.ny + w2 * v2.ny;
                    n.z = w0 * v0.nz + w1 * v1.nz + w2 * v2.nz;
                    n = v3norm(n);
                } else {
                    n = fn;
                }

                // Blinn-Phong shading
                Vec3 L = {light.dir[0], light.dir[1], light.dir[2]};
                float NdotL = clampf(v3dot(n, L), 0.0f, 1.0f);

                // Hemisphere ambient (brighter on top)
                float sky_blend = n.y * 0.5f + 0.5f;
                float amb_r = light.ambient[0] * (0.7f + 0.3f * sky_blend);
                float amb_g = light.ambient[1] * (0.7f + 0.3f * sky_blend);
                float amb_b = light.ambient[2] * (0.7f + 0.3f * sky_blend);

                // Specular (Blinn-Phong)
                Vec3 H = v3norm({L.x + view_dir.x, L.y + view_dir.y, L.z + view_dir.z});
                float NdotH = clampf(v3dot(n, H), 0.0f, 1.0f);
                float spec = std::pow(NdotH, 40.0f) * 0.35f;

                float r = mat_color[0] * (amb_r + NdotL * light.color[0]) + spec * light.color[0];
                float g = mat_color[1] * (amb_g + NdotL * light.color[1]) + spec * light.color[1];
                float b = mat_color[2] * (amb_b + NdotL * light.color[2]) + spec * light.color[2];

                r = clampf(r, 0.0f, 1.0f);
                g = clampf(g, 0.0f, 1.0f);
                b = clampf(b, 0.0f, 1.0f);

                uint8_t* p = pixels + static_cast<uint32_t>(y) * stride + static_cast<uint32_t>(x) * 4;
                p[0] = static_cast<uint8_t>(r * 255.0f);
                p[1] = static_cast<uint8_t>(g * 255.0f);
                p[2] = static_cast<uint8_t>(b * 255.0f);
                p[3] = 235;
            }
        }
    }

    delete[] sv;
    delete[] zbuf;
}

// ---------------------------------------------------------------------------
// Convenience: compute AABB from vertex data
// ---------------------------------------------------------------------------

inline void compute_aabb(const float* vert_data, uint32_t vert_count,
                          uint32_t vert_stride, uint32_t pos_offset,
                          float bmin[3], float bmax[3]) {
    if (vert_count == 0) {
        bmin[0] = bmin[1] = bmin[2] = -0.5f;
        bmax[0] = bmax[1] = bmax[2] =  0.5f;
        return;
    }
    uint32_t stride_floats = vert_stride / sizeof(float);
    uint32_t pos_off_floats = pos_offset / sizeof(float);
    const float* p = vert_data + pos_off_floats;
    bmin[0] = bmax[0] = p[0];
    bmin[1] = bmax[1] = p[1];
    bmin[2] = bmax[2] = p[2];
    for (uint32_t i = 1; i < vert_count; ++i) {
        p = vert_data + i * stride_floats + pos_off_floats;
        for (int j = 0; j < 3; ++j) {
            if (p[j] < bmin[j]) bmin[j] = p[j];
            if (p[j] > bmax[j]) bmax[j] = p[j];
        }
    }
}

} // namespace vivid::thumb3d
