#pragma once
#include "operator_api/gpu_3d.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace vivid::gpu {

class MeshBuilderUtil {
public:
    // Add vertex with default tangent {0,0,0,1}, return index.
    uint32_t addVertex(float px, float py, float pz,
                       float nx, float ny, float nz,
                       float u, float v) {
        Vertex3D vert{};
        vert.position[0] = px; vert.position[1] = py; vert.position[2] = pz;
        vert.normal[0] = nx;   vert.normal[1] = ny;   vert.normal[2] = nz;
        vert.tangent[0] = 0.f; vert.tangent[1] = 0.f; vert.tangent[2] = 0.f; vert.tangent[3] = 1.f;
        vert.uv[0] = u;        vert.uv[1] = v;
        uint32_t idx = static_cast<uint32_t>(vertices_.size());
        vertices_.push_back(vert);
        return idx;
    }

    // Add vertex with full tangent data.
    uint32_t addVertex(const Vertex3D& vert) {
        uint32_t idx = static_cast<uint32_t>(vertices_.size());
        vertices_.push_back(vert);
        return idx;
    }

    void addTriangle(uint32_t a, uint32_t b, uint32_t c) {
        indices_.push_back(a);
        indices_.push_back(b);
        indices_.push_back(c);
    }

    void addQuad(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        addTriangle(a, b, c);
        addTriangle(a, c, d);
    }

    // Compute smooth normals from face normals (area-weighted accumulation).
    void computeNormals() {
        // Zero all normals
        for (auto& v : vertices_) {
            v.normal[0] = 0.f; v.normal[1] = 0.f; v.normal[2] = 0.f;
        }
        // Accumulate face normals
        for (size_t i = 0; i + 2 < indices_.size(); i += 3) {
            Vertex3D& v0 = vertices_[indices_[i]];
            Vertex3D& v1 = vertices_[indices_[i + 1]];
            Vertex3D& v2 = vertices_[indices_[i + 2]];
            float e1[3] = { v1.position[0] - v0.position[0],
                            v1.position[1] - v0.position[1],
                            v1.position[2] - v0.position[2] };
            float e2[3] = { v2.position[0] - v0.position[0],
                            v2.position[1] - v0.position[1],
                            v2.position[2] - v0.position[2] };
            float fn[3] = { e1[1]*e2[2] - e1[2]*e2[1],
                            e1[2]*e2[0] - e1[0]*e2[2],
                            e1[0]*e2[1] - e1[1]*e2[0] };
            for (int k = 0; k < 3; ++k) {
                v0.normal[k] += fn[k];
                v1.normal[k] += fn[k];
                v2.normal[k] += fn[k];
            }
        }
        // Normalize
        for (auto& v : vertices_) {
            float len = std::sqrt(v.normal[0]*v.normal[0] + v.normal[1]*v.normal[1] + v.normal[2]*v.normal[2]);
            if (len > 1e-8f) {
                v.normal[0] /= len; v.normal[1] /= len; v.normal[2] /= len;
            }
        }
    }

    // Compute tangents (MikkTSpace-style approximation from UV deltas).
    void computeTangents() {
        // Zero tangent xyz, preserve w
        for (auto& v : vertices_) {
            v.tangent[0] = 0.f; v.tangent[1] = 0.f; v.tangent[2] = 0.f;
        }
        for (size_t i = 0; i + 2 < indices_.size(); i += 3) {
            Vertex3D& v0 = vertices_[indices_[i]];
            Vertex3D& v1 = vertices_[indices_[i + 1]];
            Vertex3D& v2 = vertices_[indices_[i + 2]];

            float e1[3] = { v1.position[0] - v0.position[0],
                            v1.position[1] - v0.position[1],
                            v1.position[2] - v0.position[2] };
            float e2[3] = { v2.position[0] - v0.position[0],
                            v2.position[1] - v0.position[1],
                            v2.position[2] - v0.position[2] };

            float du1 = v1.uv[0] - v0.uv[0], dv1 = v1.uv[1] - v0.uv[1];
            float du2 = v2.uv[0] - v0.uv[0], dv2 = v2.uv[1] - v0.uv[1];

            float det = du1 * dv2 - du2 * dv1;
            float r = (std::abs(det) > 1e-8f) ? (1.f / det) : 0.f;

            float t[3] = { r * (dv2 * e1[0] - dv1 * e2[0]),
                           r * (dv2 * e1[1] - dv1 * e2[1]),
                           r * (dv2 * e1[2] - dv1 * e2[2]) };

            for (int k = 0; k < 3; ++k) {
                v0.tangent[k] += t[k];
                v1.tangent[k] += t[k];
                v2.tangent[k] += t[k];
            }
        }
        // Normalize and set w=1 (right-handed)
        for (auto& v : vertices_) {
            float len = std::sqrt(v.tangent[0]*v.tangent[0] + v.tangent[1]*v.tangent[1] + v.tangent[2]*v.tangent[2]);
            if (len > 1e-8f) {
                v.tangent[0] /= len; v.tangent[1] /= len; v.tangent[2] /= len;
            }
            v.tangent[3] = 1.f;
        }
    }

    // Transform: translate all vertex positions.
    void translate(float x, float y, float z) {
        for (auto& v : vertices_) {
            v.position[0] += x; v.position[1] += y; v.position[2] += z;
        }
    }

    // Transform: scale positions and inverse-scale normals.
    void scale(float x, float y, float z) {
        for (auto& v : vertices_) {
            v.position[0] *= x; v.position[1] *= y; v.position[2] *= z;
            // Normals transform by inverse scale
            if (std::abs(x) > 1e-8f) v.normal[0] /= x;
            if (std::abs(y) > 1e-8f) v.normal[1] /= y;
            if (std::abs(z) > 1e-8f) v.normal[2] /= z;
            float len = std::sqrt(v.normal[0]*v.normal[0] + v.normal[1]*v.normal[1] + v.normal[2]*v.normal[2]);
            if (len > 1e-8f) {
                v.normal[0] /= len; v.normal[1] /= len; v.normal[2] /= len;
            }
        }
    }

    // Transform: rotate using Rodrigues' formula around arbitrary axis.
    void rotate(float angle_rad, float ax, float ay, float az) {
        // Normalize axis
        float alen = std::sqrt(ax*ax + ay*ay + az*az);
        if (alen < 1e-8f) return;
        ax /= alen; ay /= alen; az /= alen;

        float c = std::cos(angle_rad), s = std::sin(angle_rad);
        float t = 1.f - c;

        auto rot = [&](float x, float y, float z, float out[3]) {
            // Rodrigues: v' = v*cos(a) + (k x v)*sin(a) + k*(k.v)*(1-cos(a))
            float dot = ax*x + ay*y + az*z;
            float cx = ay*z - az*y, cy = az*x - ax*z, cz = ax*y - ay*x;
            out[0] = x*c + cx*s + ax*dot*t;
            out[1] = y*c + cy*s + ay*dot*t;
            out[2] = z*c + cz*s + az*dot*t;
        };

        for (auto& v : vertices_) {
            float p[3], n[3];
            rot(v.position[0], v.position[1], v.position[2], p);
            rot(v.normal[0], v.normal[1], v.normal[2], n);
            std::memcpy(v.position, p, sizeof(p));
            std::memcpy(v.normal, n, sizeof(n));
            // Also rotate tangent direction
            float tg[3];
            rot(v.tangent[0], v.tangent[1], v.tangent[2], tg);
            v.tangent[0] = tg[0]; v.tangent[1] = tg[1]; v.tangent[2] = tg[2];
            // w (handedness) unchanged
        }
    }

    // Merge another mesh into this one.
    void append(const MeshBuilderUtil& other) {
        uint32_t offset = static_cast<uint32_t>(vertices_.size());
        vertices_.insert(vertices_.end(), other.vertices_.begin(), other.vertices_.end());
        for (uint32_t idx : other.indices_) {
            indices_.push_back(idx + offset);
        }
    }

    void clear() {
        vertices_.clear();
        indices_.clear();
    }

    uint32_t vertexCount() const { return static_cast<uint32_t>(vertices_.size()); }
    uint32_t indexCount()  const { return static_cast<uint32_t>(indices_.size()); }

    const std::vector<Vertex3D>& vertices() const { return vertices_; }
    const std::vector<uint32_t>& indices()  const { return indices_; }

    // =========================================================================
    // Static primitive generators
    // =========================================================================

    // Box: 24 vertices (4 per face, flat normals), 36 indices.
    static MeshBuilderUtil box(float w, float h, float d) {
        MeshBuilderUtil m;
        float hw = w * 0.5f, hh = h * 0.5f, hd = d * 0.5f;

        // Face data: normal, tangent, 4 corner positions, UV coords
        struct Face {
            float nx, ny, nz;
            float tx, ty, tz, tw;
            float p[4][3];
        };
        Face faces[6] = {
            // +Z face
            { 0, 0, 1,  1, 0, 0, 1,
              {{ -hw,-hh, hd}, { hw,-hh, hd}, { hw, hh, hd}, {-hw, hh, hd}} },
            // -Z face
            { 0, 0,-1, -1, 0, 0, 1,
              {{  hw,-hh,-hd}, {-hw,-hh,-hd}, {-hw, hh,-hd}, { hw, hh,-hd}} },
            // +Y face
            { 0, 1, 0,  1, 0, 0, 1,
              {{ -hw, hh, hd}, { hw, hh, hd}, { hw, hh,-hd}, {-hw, hh,-hd}} },
            // -Y face
            { 0,-1, 0,  1, 0, 0, 1,
              {{ -hw,-hh,-hd}, { hw,-hh,-hd}, { hw,-hh, hd}, {-hw,-hh, hd}} },
            // +X face
            { 1, 0, 0,  0, 0,-1, 1,
              {{  hw,-hh, hd}, { hw,-hh,-hd}, { hw, hh,-hd}, { hw, hh, hd}} },
            // -X face
            {-1, 0, 0,  0, 0, 1, 1,
              {{ -hw,-hh,-hd}, {-hw,-hh, hd}, {-hw, hh, hd}, {-hw, hh,-hd}} },
        };
        float uvs[4][2] = {{0,1},{1,1},{1,0},{0,0}};

        for (int f = 0; f < 6; ++f) {
            uint32_t base = m.vertexCount();
            for (int c = 0; c < 4; ++c) {
                Vertex3D v{};
                v.position[0] = faces[f].p[c][0];
                v.position[1] = faces[f].p[c][1];
                v.position[2] = faces[f].p[c][2];
                v.normal[0] = faces[f].nx;
                v.normal[1] = faces[f].ny;
                v.normal[2] = faces[f].nz;
                v.tangent[0] = faces[f].tx;
                v.tangent[1] = faces[f].ty;
                v.tangent[2] = faces[f].tz;
                v.tangent[3] = faces[f].tw;
                v.uv[0] = uvs[c][0];
                v.uv[1] = uvs[c][1];
                m.addVertex(v);
            }
            m.addQuad(base, base+1, base+2, base+3);
        }
        return m;
    }

    // UV Sphere: (segments+1) latitude rings x (segments+1) longitude.
    static MeshBuilderUtil sphere(float radius, int segments = 16) {
        MeshBuilderUtil m;
        const float PI = 3.14159265358979f;

        for (int i = 0; i <= segments; ++i) {
            float phi = static_cast<float>(i) / static_cast<float>(segments) * PI;
            float sin_phi = std::sin(phi);
            float cos_phi = std::cos(phi);
            float v_coord = static_cast<float>(i) / static_cast<float>(segments);

            for (int j = 0; j <= segments; ++j) {
                float theta = static_cast<float>(j) / static_cast<float>(segments) * 2.f * PI;
                float sin_theta = std::sin(theta);
                float cos_theta = std::cos(theta);
                float u_coord = static_cast<float>(j) / static_cast<float>(segments);

                float nx = sin_phi * cos_theta;
                float ny = cos_phi;
                float nz = sin_phi * sin_theta;

                // Tangent: derivative of position w.r.t. theta
                float tx = -sin_theta;
                float tz =  cos_theta;
                float ty = 0.f;
                float tlen = std::sqrt(tx*tx + ty*ty + tz*tz);
                if (tlen > 1e-8f) { tx /= tlen; ty /= tlen; tz /= tlen; }

                Vertex3D v{};
                v.position[0] = nx * radius;
                v.position[1] = ny * radius;
                v.position[2] = nz * radius;
                v.normal[0] = nx; v.normal[1] = ny; v.normal[2] = nz;
                v.tangent[0] = tx; v.tangent[1] = ty; v.tangent[2] = tz; v.tangent[3] = 1.f;
                v.uv[0] = u_coord; v.uv[1] = v_coord;
                m.addVertex(v);
            }
        }

        int cols = segments + 1;
        for (int i = 0; i < segments; ++i) {
            for (int j = 0; j < segments; ++j) {
                uint32_t tl = static_cast<uint32_t>(i * cols + j);
                uint32_t tr = tl + 1;
                uint32_t bl = tl + static_cast<uint32_t>(cols);
                uint32_t br = bl + 1;
                m.addTriangle(tl, bl, br);
                m.addTriangle(tl, br, tr);
            }
        }
        return m;
    }

    // Cylinder: Y-axis aligned, side body + top/bottom caps.
    static MeshBuilderUtil cylinder(float radius, float height, int segments = 16) {
        MeshBuilderUtil m;
        const float PI = 3.14159265358979f;
        float hh = height * 0.5f;

        // --- Side body ---
        for (int i = 0; i <= 1; ++i) {
            float y = (i == 0) ? -hh : hh;
            float v_coord = static_cast<float>(i);
            for (int j = 0; j <= segments; ++j) {
                float theta = static_cast<float>(j) / static_cast<float>(segments) * 2.f * PI;
                float cs = std::cos(theta), sn = std::sin(theta);
                float u_coord = static_cast<float>(j) / static_cast<float>(segments);

                // Tangent along circumference
                float tx = -sn, tz = cs;

                Vertex3D v{};
                v.position[0] = cs * radius; v.position[1] = y; v.position[2] = sn * radius;
                v.normal[0] = cs; v.normal[1] = 0.f; v.normal[2] = sn;
                v.tangent[0] = tx; v.tangent[1] = 0.f; v.tangent[2] = tz; v.tangent[3] = 1.f;
                v.uv[0] = u_coord; v.uv[1] = v_coord;
                m.addVertex(v);
            }
        }
        int cols = segments + 1;
        for (int j = 0; j < segments; ++j) {
            uint32_t bl = static_cast<uint32_t>(j);
            uint32_t br = bl + 1;
            uint32_t tl = static_cast<uint32_t>(cols + j);
            uint32_t tr = tl + 1;
            m.addTriangle(bl, br, tr);
            m.addTriangle(bl, tr, tl);
        }

        // --- Top cap (Y = +hh) ---
        {
            uint32_t center = m.addVertex(0.f, hh, 0.f, 0.f, 1.f, 0.f, 0.5f, 0.5f);
            uint32_t base = m.vertexCount();
            for (int j = 0; j <= segments; ++j) {
                float theta = static_cast<float>(j) / static_cast<float>(segments) * 2.f * PI;
                float cs = std::cos(theta), sn = std::sin(theta);
                Vertex3D v{};
                v.position[0] = cs * radius; v.position[1] = hh; v.position[2] = sn * radius;
                v.normal[0] = 0.f; v.normal[1] = 1.f; v.normal[2] = 0.f;
                v.tangent[0] = 1.f; v.tangent[1] = 0.f; v.tangent[2] = 0.f; v.tangent[3] = 1.f;
                v.uv[0] = cs * 0.5f + 0.5f; v.uv[1] = sn * 0.5f + 0.5f;
                m.addVertex(v);
            }
            for (int j = 0; j < segments; ++j) {
                m.addTriangle(center, base + static_cast<uint32_t>(j + 1), base + static_cast<uint32_t>(j));
            }
        }

        // --- Bottom cap (Y = -hh) ---
        {
            uint32_t center = m.addVertex(0.f, -hh, 0.f, 0.f, -1.f, 0.f, 0.5f, 0.5f);
            uint32_t base = m.vertexCount();
            for (int j = 0; j <= segments; ++j) {
                float theta = static_cast<float>(j) / static_cast<float>(segments) * 2.f * PI;
                float cs = std::cos(theta), sn = std::sin(theta);
                Vertex3D v{};
                v.position[0] = cs * radius; v.position[1] = -hh; v.position[2] = sn * radius;
                v.normal[0] = 0.f; v.normal[1] = -1.f; v.normal[2] = 0.f;
                v.tangent[0] = 1.f; v.tangent[1] = 0.f; v.tangent[2] = 0.f; v.tangent[3] = 1.f;
                v.uv[0] = cs * 0.5f + 0.5f; v.uv[1] = sn * 0.5f + 0.5f;
                m.addVertex(v);
            }
            for (int j = 0; j < segments; ++j) {
                m.addTriangle(center, base + static_cast<uint32_t>(j), base + static_cast<uint32_t>(j + 1));
            }
        }

        return m;
    }

    // Cone: apex at (0, height/2, 0), base at Y = -height/2. Bottom cap only.
    static MeshBuilderUtil cone(float radius, float height, int segments = 16) {
        MeshBuilderUtil m;
        const float PI = 3.14159265358979f;
        float hh = height * 0.5f;

        // Slope angle for normals: the normal tilts outward by atan(radius/height)
        float slope = radius / height;
        float ny_side = 1.f / std::sqrt(1.f + slope * slope);
        float nr_side = slope * ny_side;  // radial component

        // --- Side body ---
        // Apex vertex per segment (unique normals per triangle strip)
        uint32_t base_ring_start = m.vertexCount();
        for (int j = 0; j <= segments; ++j) {
            float theta = static_cast<float>(j) / static_cast<float>(segments) * 2.f * PI;
            float cs = std::cos(theta), sn = std::sin(theta);
            float u_coord = static_cast<float>(j) / static_cast<float>(segments);

            float nx = cs * nr_side, nz = sn * nr_side;
            float tx = -sn, tz = cs;

            // Base vertex
            Vertex3D vb{};
            vb.position[0] = cs * radius; vb.position[1] = -hh; vb.position[2] = sn * radius;
            vb.normal[0] = nx; vb.normal[1] = ny_side; vb.normal[2] = nz;
            vb.tangent[0] = tx; vb.tangent[1] = 0.f; vb.tangent[2] = tz; vb.tangent[3] = 1.f;
            vb.uv[0] = u_coord; vb.uv[1] = 1.f;
            m.addVertex(vb);

            // Apex vertex (duplicated per segment for correct normals)
            Vertex3D va{};
            va.position[0] = 0.f; va.position[1] = hh; va.position[2] = 0.f;
            va.normal[0] = nx; va.normal[1] = ny_side; va.normal[2] = nz;
            va.tangent[0] = tx; va.tangent[1] = 0.f; va.tangent[2] = tz; va.tangent[3] = 1.f;
            va.uv[0] = u_coord; va.uv[1] = 0.f;
            m.addVertex(va);
        }

        for (int j = 0; j < segments; ++j) {
            uint32_t b0 = base_ring_start + static_cast<uint32_t>(j * 2);
            uint32_t a0 = b0 + 1;
            uint32_t b1 = b0 + 2;
            m.addTriangle(b0, b1, a0);
        }

        // --- Bottom cap ---
        {
            uint32_t center = m.addVertex(0.f, -hh, 0.f, 0.f, -1.f, 0.f, 0.5f, 0.5f);
            uint32_t cap_base = m.vertexCount();
            for (int j = 0; j <= segments; ++j) {
                float theta = static_cast<float>(j) / static_cast<float>(segments) * 2.f * PI;
                float cs = std::cos(theta), sn = std::sin(theta);
                Vertex3D v{};
                v.position[0] = cs * radius; v.position[1] = -hh; v.position[2] = sn * radius;
                v.normal[0] = 0.f; v.normal[1] = -1.f; v.normal[2] = 0.f;
                v.tangent[0] = 1.f; v.tangent[1] = 0.f; v.tangent[2] = 0.f; v.tangent[3] = 1.f;
                v.uv[0] = cs * 0.5f + 0.5f; v.uv[1] = sn * 0.5f + 0.5f;
                m.addVertex(v);
            }
            for (int j = 0; j < segments; ++j) {
                m.addTriangle(center, cap_base + static_cast<uint32_t>(j), cap_base + static_cast<uint32_t>(j + 1));
            }
        }

        return m;
    }

    // Torus: XZ plane, major radius = outer_radius, tube radius = inner_radius.
    static MeshBuilderUtil torus(float outer_radius, float inner_radius,
                                  int segments = 16, int rings = 16) {
        MeshBuilderUtil m;
        const float PI = 3.14159265358979f;

        for (int i = 0; i <= rings; ++i) {
            float phi = static_cast<float>(i) / static_cast<float>(rings) * 2.f * PI;
            float cp = std::cos(phi), sp = std::sin(phi);

            for (int j = 0; j <= segments; ++j) {
                float theta = static_cast<float>(j) / static_cast<float>(segments) * 2.f * PI;
                float ct = std::cos(theta), st = std::sin(theta);

                // Center of tube ring
                float cx = outer_radius * cp;
                float cz = outer_radius * sp;

                // Position on tube surface
                float px = (outer_radius + inner_radius * ct) * cp;
                float py = inner_radius * st;
                float pz = (outer_radius + inner_radius * ct) * sp;

                // Normal: from ring center to surface point
                float nx = ct * cp;
                float ny = st;
                float nz = ct * sp;

                // Tangent: along the ring (derivative w.r.t. phi)
                float tx = -sp;
                float tz =  cp;

                Vertex3D v{};
                v.position[0] = px; v.position[1] = py; v.position[2] = pz;
                v.normal[0] = nx; v.normal[1] = ny; v.normal[2] = nz;
                v.tangent[0] = tx; v.tangent[1] = 0.f; v.tangent[2] = tz; v.tangent[3] = 1.f;
                v.uv[0] = static_cast<float>(j) / static_cast<float>(segments);
                v.uv[1] = static_cast<float>(i) / static_cast<float>(rings);
                m.addVertex(v);
            }
        }

        int cols = segments + 1;
        for (int i = 0; i < rings; ++i) {
            for (int j = 0; j < segments; ++j) {
                uint32_t tl = static_cast<uint32_t>(i * cols + j);
                uint32_t tr = tl + 1;
                uint32_t bl = tl + static_cast<uint32_t>(cols);
                uint32_t br = bl + 1;
                m.addTriangle(tl, bl, br);
                m.addTriangle(tl, br, tr);
            }
        }
        return m;
    }

    // Plane: XZ plane at Y=0, subdivided.
    static MeshBuilderUtil plane(float w, float h, int subdX = 1, int subdY = 1) {
        MeshBuilderUtil m;
        float hw = w * 0.5f, hh = h * 0.5f;

        for (int iy = 0; iy <= subdY; ++iy) {
            float fy = static_cast<float>(iy) / static_cast<float>(subdY);
            float z = -hh + fy * h;
            for (int ix = 0; ix <= subdX; ++ix) {
                float fx = static_cast<float>(ix) / static_cast<float>(subdX);
                float x = -hw + fx * w;
                Vertex3D v{};
                v.position[0] = x; v.position[1] = 0.f; v.position[2] = z;
                v.normal[0] = 0.f; v.normal[1] = 1.f; v.normal[2] = 0.f;
                v.tangent[0] = 1.f; v.tangent[1] = 0.f; v.tangent[2] = 0.f; v.tangent[3] = 1.f;
                v.uv[0] = fx; v.uv[1] = fy;
                m.addVertex(v);
            }
        }

        int cols = subdX + 1;
        for (int iy = 0; iy < subdY; ++iy) {
            for (int ix = 0; ix < subdX; ++ix) {
                uint32_t tl = static_cast<uint32_t>(iy * cols + ix);
                uint32_t tr = tl + 1;
                uint32_t bl = tl + static_cast<uint32_t>(cols);
                uint32_t br = bl + 1;
                m.addQuad(tl, bl, br, tr);
            }
        }
        return m;
    }

    // Upload to GPU using gpu_3d.h helpers.
    struct GpuBuffers {
        WGPUBuffer vertex_buffer;
        WGPUBuffer index_buffer;
        uint32_t index_count;
        uint64_t vertex_buf_size;
    };

    GpuBuffers uploadBuffers(WGPUDevice device, WGPUQueue queue,
                              const char* label = "MeshBuilder") const {
        GpuBuffers result{};
        uint64_t vbytes = vertices_.size() * sizeof(Vertex3D);
        uint64_t icount = indices_.size();

        // Build labels
        char vl[128], il[128];
        std::snprintf(vl, sizeof(vl), "%s VB", label);
        std::snprintf(il, sizeof(il), "%s IB", label);

        result.vertex_buffer  = create_vertex_buffer(device, queue, vertices_.data(), vbytes, vl);
        result.index_buffer   = create_index_buffer(device, queue, indices_.data(), icount, il);
        result.index_count    = static_cast<uint32_t>(icount);
        result.vertex_buf_size = vbytes;
        return result;
    }

private:
    std::vector<Vertex3D> vertices_;
    std::vector<uint32_t> indices_;
};

} // namespace vivid::gpu
