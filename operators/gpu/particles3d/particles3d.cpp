#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include "operator_api/draw_ui_helpers.h"
#include "operator_api/editor_ui.h"
#include "operator_api/editor_keys.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// =============================================================================
// Particles3D — GPU compute particle simulation with billboard rendering
// =============================================================================

// ---------------------------------------------------------------------------
// Compute shader: particle simulation + instance data generation
// ---------------------------------------------------------------------------

static const char* kParticlesCompute = R"(
struct Particle {
    position: vec3f,
    age: f32,
    velocity: vec3f,
    lifetime: f32,
}

struct Params {
    max_count: u32,
    new_spawns: u32,
    dt: f32,
    gravity: f32,
    speed: f32,
    spread_rad: f32,
    lifetime: f32,
    size: f32,
    color: vec4f,
    seed: u32,
    noise_octaves: u32,
    noise_scale: f32,
    noise_speed: f32,
    curl_strength: f32,
    drag: f32,
    time: f32,
    elongation: f32,
    shape: u32,
    learning_mode: u32, // 0=Advanced, 1=Beginner
    bounds: f32,
    _pad0: f32,
}

struct InstanceData {
    pos_rot:   vec4f,   // xyz=position, w=rotation_y (yaw)
    scale_pad: vec4f,   // xyz=scale, w=rotation_x (pitch)
    color:     vec4f,
}

@group(0) @binding(0) var<storage, read>       particles_in:  array<Particle>;
@group(0) @binding(1) var<storage, read_write>  particles_out: array<Particle>;
@group(0) @binding(2) var<storage, read_write>  instances_out: array<InstanceData>;
@group(0) @binding(3) var<uniform>              params: Params;
@group(0) @binding(4) var<storage, read_write>  counter: atomic<u32>;

// --- Noise functions for curl noise ---

fn permute(x: vec4f) -> vec4f {
    return (((x * 34.0) + 1.0) * x) % 289.0;
}

fn taylorInvSqrt(r: vec4f) -> vec4f {
    return 1.79284291400159 - 0.85373472095314 * r;
}

fn simplex3D(v: vec3f) -> f32 {
    let C = vec2f(1.0/6.0, 1.0/3.0);
    let D = vec4f(0.0, 0.5, 1.0, 2.0);

    var i = floor(v + dot(v, C.yyy));
    let x0 = v - i + dot(i, C.xxx);

    let g = step(x0.yzx, x0.xyz);
    let l = 1.0 - g;
    let i1 = min(g.xyz, l.zxy);
    let i2 = max(g.xyz, l.zxy);

    let x1 = x0 - i1 + C.xxx;
    let x2 = x0 - i2 + C.yyy;
    let x3 = x0 - D.yyy;

    i = i % 289.0;
    let p = permute(permute(permute(
             i.z + vec4f(0.0, i1.z, i2.z, 1.0))
           + i.y + vec4f(0.0, i1.y, i2.y, 1.0))
           + i.x + vec4f(0.0, i1.x, i2.x, 1.0));

    let n_ = 0.142857142857;
    let ns = n_ * D.wyz - D.xzx;

    let j = p - 49.0 * floor(p * ns.z * ns.z);

    let x_ = floor(j * ns.z);
    let y_ = floor(j - 7.0 * x_);

    let x = x_ * ns.x + ns.yyyy;
    let y = y_ * ns.x + ns.yyyy;
    let h = 1.0 - abs(x) - abs(y);

    let b0 = vec4f(x.xy, y.xy);
    let b1 = vec4f(x.zw, y.zw);

    let s0 = floor(b0) * 2.0 + 1.0;
    let s1 = floor(b1) * 2.0 + 1.0;
    let sh = -step(h, vec4f(0.0));

    let a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    let a1 = b1.xzyw + s1.xzyw * sh.zzww;

    var p0 = vec3f(a0.xy, h.x);
    var p1 = vec3f(a0.zw, h.y);
    var p2 = vec3f(a1.xy, h.z);
    var p3 = vec3f(a1.zw, h.w);

    let norm = taylorInvSqrt(vec4f(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 = p0 * norm.x;
    p1 = p1 * norm.y;
    p2 = p2 * norm.z;
    p3 = p3 * norm.w;

    var m = max(vec4f(0.5) - vec4f(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), vec4f(0.0));
    m = m * m;
    return dot(m * m, vec4f(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3))) * 105.0;
}

fn fbm_simplex3D(p_in: vec3f, octaves: u32, lacunarity: f32, persistence: f32) -> f32 {
    var value = 0.0;
    var amplitude = 1.0;
    var frequency = 1.0;
    var max_value = 0.0;
    var p = p_in;
    for (var i = 0u; i < octaves; i++) {
        value += amplitude * simplex3D(p * frequency);
        max_value += amplitude;
        amplitude *= persistence;
        frequency *= lacunarity;
    }
    return value / max_value;
}

fn curl_noise(p: vec3f, octaves: u32) -> vec3f {
    let e = 0.01;
    // Finite differences for curl of a potential field
    // curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)
    // We use 3 independent noise fields offset by large constants
    let fx_py = fbm_simplex3D(p + vec3f(0.0, e, 0.0), octaves, 2.0, 0.5);
    let fx_ny = fbm_simplex3D(p - vec3f(0.0, e, 0.0), octaves, 2.0, 0.5);
    let fx_pz = fbm_simplex3D(p + vec3f(0.0, 0.0, e), octaves, 2.0, 0.5);
    let fx_nz = fbm_simplex3D(p - vec3f(0.0, 0.0, e), octaves, 2.0, 0.5);

    let fy_px = fbm_simplex3D(p + vec3f(e, 0.0, 0.0) + vec3f(123.4, 0.0, 0.0), octaves, 2.0, 0.5);
    let fy_nx = fbm_simplex3D(p - vec3f(e, 0.0, 0.0) + vec3f(123.4, 0.0, 0.0), octaves, 2.0, 0.5);
    let fy_pz = fbm_simplex3D(p + vec3f(0.0, 0.0, e) + vec3f(123.4, 0.0, 0.0), octaves, 2.0, 0.5);
    let fy_nz = fbm_simplex3D(p - vec3f(0.0, 0.0, e) + vec3f(123.4, 0.0, 0.0), octaves, 2.0, 0.5);

    let fz_px = fbm_simplex3D(p + vec3f(e, 0.0, 0.0) + vec3f(0.0, 456.7, 0.0), octaves, 2.0, 0.5);
    let fz_nx = fbm_simplex3D(p - vec3f(e, 0.0, 0.0) + vec3f(0.0, 456.7, 0.0), octaves, 2.0, 0.5);
    let fz_py = fbm_simplex3D(p + vec3f(0.0, e, 0.0) + vec3f(0.0, 456.7, 0.0), octaves, 2.0, 0.5);
    let fz_ny = fbm_simplex3D(p - vec3f(0.0, e, 0.0) + vec3f(0.0, 456.7, 0.0), octaves, 2.0, 0.5);

    let inv2e = 1.0 / (2.0 * e);
    return vec3f(
        (fz_py - fz_ny) * inv2e - (fy_pz - fy_nz) * inv2e,
        (fx_pz - fx_nz) * inv2e - (fz_px - fz_nx) * inv2e,
        (fy_px - fy_nx) * inv2e - (fx_py - fx_ny) * inv2e
    );
}

// PCG hash — fast deterministic PRNG
fn pcg_hash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_float(seed: u32) -> f32 {
    return f32(pcg_hash(seed)) / 4294967295.0;
}

@compute @workgroup_size(256)
fn cs_main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.max_count) { return; }

    var p = particles_in[idx];
    let is_dead = p.lifetime <= 0.0;

    if (is_dead) {
        // Try to claim a spawn slot
        let slot = atomicAdd(&counter, 1u);
        if (slot < params.new_spawns) {
            // Initialize new particle at origin with random velocity in emission cone
            let s0 = pcg_hash(params.seed + idx * 3u);
            let s1 = pcg_hash(s0);
            let s2 = pcg_hash(s1);

            let phi = rand_float(s0) * 6.28318530718;
            let cos_theta_min = cos(params.spread_rad);
            let cos_theta = cos_theta_min + rand_float(s1) * (1.0 - cos_theta_min);
            let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            let vx = sin_theta * cos(phi) * params.speed;
            let vy = cos_theta * params.speed;
            let vz = sin_theta * sin(phi) * params.speed;

            p.position = vec3f(0.0, 0.0, 0.0);
            p.velocity = vec3f(vx, vy, vz);
            p.age = 0.0;
            p.lifetime = params.lifetime * (0.8 + 0.4 * rand_float(s2));
        } else {
            // Stay dead
            p.lifetime = 0.0;
        }
    } else {
        // Integrate live particle
        p.velocity.y += params.gravity * params.dt;

        // Curl noise force
        if (params.curl_strength > 0.0) {
            let noise_pos = p.position * params.noise_scale
                          + vec3f(0.0, 0.0, params.time * params.noise_speed);
            let curl_force = curl_noise(noise_pos, params.noise_octaves);
            p.velocity += curl_force * params.curl_strength * params.dt;
        }
        // Drag
        if (params.drag > 0.0) {
            p.velocity *= 1.0 - params.drag * params.dt;
        }

        p.position += p.velocity * params.dt;
        // Beginner-friendly safety bounds: out-of-range particles are reset.
        if (params.bounds > 0.0) {
            let out_of_bounds = abs(p.position.x) > params.bounds
                             || abs(p.position.y) > params.bounds
                             || abs(p.position.z) > params.bounds;
            if (out_of_bounds) {
                p.lifetime = 0.0;
            }
        }
        p.age += params.dt;
        if (p.age >= p.lifetime) {
            p.lifetime = 0.0;  // kill
        }
    }

    particles_out[idx] = p;

    // Write instance data
    var inst: InstanceData;
    if (p.lifetime > 0.0) {
        let age_ratio = p.age / p.lifetime;
        let size_factor = 1.0 - age_ratio * age_ratio;  // shrink over lifetime
        let alpha_factor = 1.0 - age_ratio;              // fade over lifetime
        let sz = params.size * size_factor;

        if (params.shape == 1u) {
            // Cuboid: orient along velocity vector
            let vel_len = length(p.velocity);
            let dir = select(vec3f(0.0, 1.0, 0.0), p.velocity / vel_len, vel_len > 0.001);
            let yaw = atan2(dir.x, dir.z);
            let pitch = -asin(clamp(dir.y, -1.0, 1.0));
            inst.pos_rot = vec4f(p.position, yaw);
            inst.scale_pad = vec4f(sz, sz, sz * params.elongation, pitch);
        } else {
            // Billboard: uniform scale, no rotation
            inst.pos_rot = vec4f(p.position, 0.0);
            inst.scale_pad = vec4f(sz, sz, sz, 0.0);
        }
        inst.color = vec4f(params.color.rgb, params.color.a * alpha_factor);
    } else {
        // Dead: zero-scale far away
        inst.pos_rot = vec4f(99999.0, 99999.0, 99999.0, 0.0);
        inst.scale_pad = vec4f(0.0, 0.0, 0.0, 0.0);
        inst.color = vec4f(0.0, 0.0, 0.0, 0.0);
    }
    instances_out[idx] = inst;
}
)";

// ---------------------------------------------------------------------------
// Params uniform (CPU-side, maps to GPU struct)
// ---------------------------------------------------------------------------

struct ParamsData {
    uint32_t max_count;       // 0
    uint32_t new_spawns;      // 4
    float    dt;              // 8
    float    gravity;         // 12
    float    speed;           // 16
    float    spread_rad;      // 20
    float    lifetime;        // 24
    float    size;            // 28
    float    color[4];        // 32
    uint32_t seed;            // 48
    uint32_t noise_octaves;   // 52
    float    noise_scale;     // 56
    float    noise_speed;     // 60
    float    curl_strength;   // 64
    float    drag;            // 68
    float    time;            // 72
    float    elongation;      // 76
    uint32_t shape;           // 80
    uint32_t learning_mode;   // 84
    float    bounds;          // 88
    float    _pad0;           // 92
};
static_assert(sizeof(ParamsData) == 96, "ParamsData must be 96 bytes");

// =============================================================================
// Particles3D Operator
// =============================================================================

/**
 * @brief Simulates and renders a configurable 3D particle system.
 *
 * Particles3D combines spawning, motion, curl noise, and simple shading controls to produce a
 * self-contained GPU particle field for atmospheric scenes and reactive effects.
 *
 * @param count Maximum live particle count.
 * @param emission_rate Rate at which new particles are emitted.
 * @param lifetime Lifetime of each particle.
 * @param speed Base particle speed.
 * @param gravity Vertical acceleration applied to particles.
 * @param learning_mode Chooses between advanced and beginner compute paths.
 */
struct Particles3D : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "Particles3D";
    static constexpr bool kTimeDependent = true;

    // Emission
    vivid::Param<int>   count         {"count",         1000, 1, 100000};
    vivid::Param<float> emission_rate {"emission_rate", 100.0f, 0.0f, 10000.0f};
    vivid::Param<float> lifetime      {"lifetime",      2.0f, 0.1f, 30.0f};

    // Physics
    vivid::Param<float> speed   {"speed",   2.0f, 0.0f, 20.0f};
    vivid::Param<float> gravity {"gravity", -2.0f, -20.0f, 20.0f};
    vivid::Param<float> spread  {"spread",  45.0f, 0.0f, 360.0f};
    vivid::Param<float> drag    {"drag",    0.0f, 0.0f, 10.0f};

    // Curl Noise
    vivid::Param<float> curl_strength  {"curl_strength",  0.0f, 0.0f, 20.0f};
    vivid::Param<float> noise_scale    {"noise_scale",    1.0f, 0.01f, 10.0f};
    vivid::Param<float> noise_speed    {"noise_speed",    0.5f, 0.0f, 5.0f};
    vivid::Param<int>   noise_octaves  {"noise_octaves",  2, 1, 4};

    // Appearance
    vivid::Param<int>   shape      {"shape",      0, {"Billboard", "Cuboid"}};
    vivid::Param<float> elongation {"elongation",  5.0f, 1.0f, 20.0f};
    vivid::Param<float> size       {"size",        0.05f, 0.01f, 2.0f};
    vivid::Param<float> bounds     {"bounds",      20.0f, 1.0f, 200.0f};

    // Color (warm orange default)
    vivid::Param<float> r {"r", 1.0f, 0.0f, 1.0f};
    vivid::Param<float> g {"g", 0.6f, 0.0f, 1.0f};
    vivid::Param<float> b {"b", 0.2f, 0.0f, 1.0f};
    vivid::Param<float> a {"a", 0.8f, 0.0f, 1.0f};

    // Material
    vivid::Param<float> emission {"emission", 0.5f, 0.0f, 5.0f};
    vivid::Param<int>   unlit    {"unlit",    1, {"Off", "On"}};

    // Learning mode
    vivid::Param<int> learning_mode {"learning_mode", 0, {"Advanced", "Beginner"}};

    // Editor UI state: one SliderState per numeric param, plus a cluster
    // collapse bitmask for future use. Public so tests can arrange.
    //
    // Slider slot ordering (matches the cluster layout in draw_editor):
    //   0..2  Emission: count, emission_rate, lifetime
    //   3..6  Physics:  speed, gravity, spread, drag
    //   7..10 Curl Noise: curl_strength, noise_scale, noise_speed, noise_octaves
    //   11..13 Appearance: elongation, size, bounds
    //   14..17 Color: r, g, b, a
    //   18     Material: emission
    static constexpr int kSliderCount = 19;
    vivid::ui::SliderState editor_sliders_[kSliderCount] = {};

    // Editor entry points (definitions live just below the struct body).
    static VividEditorMetadata editor_metadata();
    void draw_editor(VividEditorContext* ctx);

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        vivid::param_group(learning_mode, "Learning");

        vivid::param_group(count, "Emission");
        vivid::param_group(emission_rate, "Emission");
        vivid::param_group(lifetime, "Emission");

        vivid::param_group(speed, "Physics");
        vivid::param_group(gravity, "Physics");
        vivid::param_group(spread, "Physics");
        vivid::param_group(drag, "Physics");

        vivid::param_group(curl_strength, "Curl Noise");
        vivid::param_group(noise_scale, "Curl Noise");
        vivid::param_group(noise_speed, "Curl Noise");
        vivid::param_group(noise_octaves, "Curl Noise");

        vivid::param_group(shape, "Appearance");
        vivid::param_group(elongation, "Appearance");
        vivid::param_group(size, "Appearance");
        vivid::param_group(bounds, "Appearance");

        vivid::param_group(r, "Color");
        vivid::param_group(g, "Color");
        vivid::param_group(b, "Color");
        vivid::param_group(a, "Color");
        vivid::display_hint(r, VIVID_DISPLAY_COLOR);
        vivid::display_hint(g, VIVID_DISPLAY_COLOR);
        vivid::display_hint(b, VIVID_DISPLAY_COLOR);

        vivid::param_group(emission, "Material");
        vivid::param_group(unlit, "Material");

        out.push_back(&count);
        out.push_back(&emission_rate);
        out.push_back(&lifetime);
        out.push_back(&speed);
        out.push_back(&gravity);
        out.push_back(&spread);
        out.push_back(&drag);
        out.push_back(&curl_strength);
        out.push_back(&noise_scale);
        out.push_back(&noise_speed);
        out.push_back(&noise_octaves);
        out.push_back(&shape);
        out.push_back(&elongation);
        out.push_back(&size);
        out.push_back(&bounds);
        out.push_back(&r);
        out.push_back(&g);
        out.push_back(&b);
        out.push_back(&a);
        out.push_back(&emission);
        out.push_back(&unlit);
        out.push_back(&learning_mode);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_INPUT));
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_OUTPUT));
    }

    void draw_thumbnail(const VividThumbnailContext* ctx) override {
        if (!ctx || !ctx->draw.opaque) return;
        VividDrawAPI d = ctx->draw;
        void* o = d.opaque;

        float w = static_cast<float>(ctx->thumbnail_logical_width
                                         ? ctx->thumbnail_logical_width
                                         : ctx->thumbnail_width);
        float h = static_cast<float>(ctx->thumbnail_logical_height
                                         ? ctx->thumbnail_logical_height
                                         : ctx->thumbnail_height);

        vivid::draw_plot::draw_thumb_background(d, o, w, h);
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 3.0f, "Particles");

        // Badge: count with k-suffix for readability.
        int cnt = count.int_value();
        char badge[16];
        if (cnt >= 1000) std::snprintf(badge, sizeof(badge), "%dk", cnt / 1000);
        else             std::snprintf(badge, sizeof(badge), "%d", cnt);
        float bw = d.text_width ? d.text_width(o, badge, 0.8f) : 24.0f;
        vivid::draw_plot::draw_thumb_value(d, o, w - bw - 6.0f, 3.0f, bw, badge);

        // Body: a fixed pseudo-random scatter of dots — particles live GPU-side
        // and can't be read here, so we show a representative cloud instead.
        // Dot size follows the `size` param; shape (billboard vs cuboid) picks
        // square vs rounded primitives; colour is the operator's r/g/b/a.
        float body_top = 18.0f;
        float body_h = h - body_top - 4.0f;
        float body_w = w - 8.0f;
        float cx = w * 0.5f;
        float cy = body_top + body_h * 0.5f;

        // 16 fixed positions (Poisson-disc-ish by eye), in [-1, 1]^2.
        static const float kPositions[16][2] = {
            {-0.70f,  0.55f}, { 0.15f,  0.70f}, { 0.75f,  0.45f},
            {-0.85f,  0.00f}, {-0.30f,  0.25f}, { 0.40f,  0.20f},
            { 0.80f, -0.05f}, {-0.55f, -0.35f}, { 0.05f, -0.15f},
            {-0.15f, -0.60f}, { 0.55f, -0.40f}, {-0.75f, -0.70f},
            { 0.30f, -0.80f}, { 0.85f, -0.70f}, {-0.05f,  0.00f},
            { 0.20f,  0.45f},
        };

        float size_clamped = std::clamp(size.value * 8.0f, 1.2f, 4.5f);
        float rx = body_w * 0.45f;
        float ry = body_h * 0.40f;

        float cr = r.value, cg = g.value, cb = b.value, ca = std::max(0.35f, a.value);
        bool is_cuboid = shape.int_value() == 1;

        for (int i = 0; i < 16; ++i) {
            float px = cx + kPositions[i][0] * rx;
            float py = cy + kPositions[i][1] * ry;
            // Vary alpha a little to suggest motion / depth.
            float var = 0.65f + 0.35f * ((i * 37) % 100) / 100.0f;
            VividColor col = {cr, cg, cb, ca * var};
            if (is_cuboid) {
                d.draw_rounded_rect(o, px - size_clamped * 0.5f, py - size_clamped * 0.5f,
                                    size_clamped, size_clamped, 0.8f, col);
            } else {
                // Billboard — round dot.
                float r_dot = size_clamped * 0.5f;
                d.draw_rounded_rect(o, px - r_dot, py - r_dot,
                                    r_dot * 2.0f, r_dot * 2.0f, r_dot, col);
            }
        }

        // Emission halo when emission > 0: faint glow behind the cluster.
        if (emission.value > 0.05f) {
            float e = std::clamp(emission.value / 2.0f, 0.0f, 1.0f);
            d.draw_rounded_rect(o, cx - rx * 1.05f, cy - ry * 1.05f,
                                rx * 2.1f, ry * 2.1f, 8.0f,
                                VividColor{cr, cg, cb, 0.12f * e});
        }
    }

    void process_gpu(const VividGpuContext* ctx) override {
        uint32_t max_count = static_cast<uint32_t>(count.int_value());
        if (max_count == 0) max_count = 1;

        // Rebuild GPU resources if count changed
        if (max_count != current_count_ || !compute_pipeline_) {
            rebuild_gpu_resources(ctx, max_count);
        }
        if (!compute_pipeline_) return;

        float dt = static_cast<float>(ctx->delta_time);

        // Compute new spawn count with fractional accumulation
        spawn_accumulator_ += emission_rate.value * dt;
        uint32_t new_spawns = static_cast<uint32_t>(spawn_accumulator_);
        spawn_accumulator_ -= static_cast<float>(new_spawns);
        if (new_spawns > max_count) new_spawns = max_count;

        // Upload params
        ParamsData params{};
        const bool beginner = (learning_mode.int_value() == 1);
        params.max_count  = max_count;
        params.new_spawns = new_spawns;
        params.dt         = dt;
        params.gravity    = gravity.value;
        params.speed      = speed.value;
        params.spread_rad = spread.value * (3.14159265358979f / 180.0f);
        params.lifetime   = lifetime.value;
        params.size       = size.value;
        params.color[0]   = r.value;
        params.color[1]   = g.value;
        params.color[2]   = b.value;
        params.color[3]   = a.value;
        params.seed           = frame_counter_++;
        params.noise_octaves  = beginner ? 1u : static_cast<uint32_t>(noise_octaves.int_value());
        params.noise_scale    = beginner ? 1.0f : noise_scale.value;
        params.noise_speed    = beginner ? 0.0f : noise_speed.value;
        params.curl_strength  = beginner ? 0.0f : curl_strength.value;
        params.drag           = beginner ? 0.05f : drag.value;
        elapsed_time_ += dt;
        params.time           = elapsed_time_;
        params.elongation     = beginner ? 1.0f : elongation.value;
        params.shape          = beginner ? 0u : static_cast<uint32_t>(shape.int_value());
        params.learning_mode  = beginner ? 1u : 0u;
        params.bounds         = bounds.value;
        wgpuQueueWriteBuffer(ctx->queue, params_ubo_, 0, &params, sizeof(params));

        // Reset atomic counter to 0
        uint32_t zero = 0;
        wgpuQueueWriteBuffer(ctx->queue, counter_buf_, 0, &zero, sizeof(zero));

        // Run compute pass
        WGPUComputePassDescriptor cp_desc{};
        cp_desc.label = vivid_sv("Particles3D Compute");
        WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(
            ctx->command_encoder, &cp_desc);
        wgpuComputePassEncoderSetPipeline(pass, compute_pipeline_);
        wgpuComputePassEncoderSetBindGroup(pass, 0,
            ping_ ? bind_group_a_ : bind_group_b_, 0, nullptr);
        uint32_t workgroups = (max_count + 255) / 256;
        wgpuComputePassEncoderDispatchWorkgroups(pass, workgroups, 1, 1);
        wgpuComputePassEncoderEnd(pass);
        wgpuComputePassEncoderRelease(pass);

        // Swap ping-pong
        ping_ = !ping_;

        // Output scene fragment
        vivid::gpu::scene_fragment_identity(fragment_);
        fragment_.instance_buffer = instance_buf_;
        fragment_.instance_count  = max_count;
        fragment_.cast_shadow     = false;

        // Use input scene geometry if connected, otherwise fall back to built-in shapes
        auto* scene_in = vivid::gpu::scene_input(ctx, 0);
        if (scene_in && scene_in->vertex_buffer && scene_in->index_count > 0) {
            const auto* input = scene_in;
            fragment_.vertex_buffer   = input->vertex_buffer;
            fragment_.vertex_buf_size = input->vertex_buf_size;
            fragment_.index_buffer    = input->index_buffer;
            fragment_.index_count     = input->index_count;
            fragment_.billboard       = false;
            fragment_.depth_write     = true;
        } else {
            int shape_val = shape.int_value();
            if (shape_val == 1) {
                // Cuboid mesh
                fragment_.vertex_buffer   = box_vb_;
                fragment_.vertex_buf_size = 24 * sizeof(vivid::gpu::Vertex3D);
                fragment_.index_buffer    = box_ib_;
                fragment_.index_count     = 36;
                fragment_.billboard       = false;
                fragment_.depth_write     = true;
            } else {
                // Billboard quad
                fragment_.vertex_buffer   = quad_vb_;
                fragment_.vertex_buf_size = 4 * sizeof(vivid::gpu::Vertex3D);
                fragment_.index_buffer    = quad_ib_;
                fragment_.index_count     = 6;
                fragment_.billboard       = true;
                fragment_.depth_write     = false;
            }
        }

        fragment_.color[0]  = r.value;
        fragment_.color[1]  = g.value;
        fragment_.color[2]  = b.value;
        fragment_.color[3]  = a.value;
        fragment_.emission  = emission.value;
        fragment_.unlit     = unlit.int_value() != 0;
        fragment_.roughness = 0.5f;
        fragment_.metallic  = 0.0f;

        ctx->custom_outputs[0] = &fragment_;
    }

    ~Particles3D() override {
        vivid::gpu::release(compute_pipeline_);
        vivid::gpu::release(compute_shader_);
        vivid::gpu::release(compute_pipe_layout_);
        vivid::gpu::release(compute_bgl_);
        vivid::gpu::release(bind_group_a_);
        vivid::gpu::release(bind_group_b_);
        vivid::gpu::release(particle_buf_a_);
        vivid::gpu::release(particle_buf_b_);
        vivid::gpu::release(instance_buf_);
        vivid::gpu::release(params_ubo_);
        vivid::gpu::release(counter_buf_);
        vivid::gpu::release(quad_vb_);
        vivid::gpu::release(quad_ib_);
        vivid::gpu::release(box_vb_);
        vivid::gpu::release(box_ib_);
    }

private:
    vivid::gpu::VividSceneFragment fragment_{};

    // Compute pipeline
    WGPUComputePipeline  compute_pipeline_   = nullptr;
    WGPUShaderModule     compute_shader_     = nullptr;
    WGPUPipelineLayout   compute_pipe_layout_ = nullptr;
    WGPUBindGroupLayout  compute_bgl_        = nullptr;

    // Ping-pong bind groups
    WGPUBindGroup bind_group_a_ = nullptr;  // reads A, writes B
    WGPUBindGroup bind_group_b_ = nullptr;  // reads B, writes A

    // GPU buffers
    WGPUBuffer particle_buf_a_ = nullptr;
    WGPUBuffer particle_buf_b_ = nullptr;
    WGPUBuffer instance_buf_   = nullptr;
    WGPUBuffer params_ubo_     = nullptr;
    WGPUBuffer counter_buf_    = nullptr;

    // Billboard quad
    WGPUBuffer quad_vb_ = nullptr;
    WGPUBuffer quad_ib_ = nullptr;

    // Cuboid mesh
    WGPUBuffer box_vb_ = nullptr;
    WGPUBuffer box_ib_ = nullptr;

    uint32_t current_count_ = 0;
    bool     ping_          = true;
    float    spawn_accumulator_ = 0.0f;
    uint32_t frame_counter_     = 0;
    float    elapsed_time_      = 0.0f;

    void rebuild_gpu_resources(const VividGpuContext* gpu, uint32_t max_count) {
        // Release existing resources
        vivid::gpu::release(compute_pipeline_);
        vivid::gpu::release(compute_shader_);
        vivid::gpu::release(compute_pipe_layout_);
        vivid::gpu::release(compute_bgl_);
        vivid::gpu::release(bind_group_a_);
        vivid::gpu::release(bind_group_b_);
        vivid::gpu::release(particle_buf_a_);
        vivid::gpu::release(particle_buf_b_);
        vivid::gpu::release(instance_buf_);
        vivid::gpu::release(params_ubo_);
        vivid::gpu::release(counter_buf_);
        vivid::gpu::release(quad_vb_);
        vivid::gpu::release(quad_ib_);
        vivid::gpu::release(box_vb_);
        vivid::gpu::release(box_ib_);

        current_count_ = max_count;
        ping_ = true;

        // --- Particle buffers (32 bytes per particle) ---
        uint64_t particle_buf_size = static_cast<uint64_t>(max_count) * 32;
        if (particle_buf_size < 32) particle_buf_size = 32;

        auto make_storage_buf = [&](const char* label, uint64_t sz) -> WGPUBuffer {
            WGPUBufferDescriptor desc{};
            desc.label = vivid_sv(label);
            desc.size  = sz;
            desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
            return wgpuDeviceCreateBuffer(gpu->device, &desc);
        };

        particle_buf_a_ = make_storage_buf("Particles3D Buf A", particle_buf_size);
        particle_buf_b_ = make_storage_buf("Particles3D Buf B", particle_buf_size);

        // Instance buffer (48 bytes per InstanceData3D)
        uint64_t instance_buf_size = static_cast<uint64_t>(max_count) * sizeof(vivid::gpu::InstanceData3D);
        if (instance_buf_size < 48) instance_buf_size = 48;
        instance_buf_ = make_storage_buf("Particles3D Instances", instance_buf_size);

        // Zero-fill particle buffers (all particles start dead)
        std::vector<uint8_t> zeros(static_cast<size_t>(particle_buf_size), 0);
        wgpuQueueWriteBuffer(gpu->queue, particle_buf_a_, 0, zeros.data(), particle_buf_size);
        wgpuQueueWriteBuffer(gpu->queue, particle_buf_b_, 0, zeros.data(), particle_buf_size);

        // Params uniform buffer
        params_ubo_ = vivid::gpu::create_uniform_buffer(gpu->device, sizeof(ParamsData), "Particles3D Params");

        // Counter buffer (atomic<u32>)
        counter_buf_ = make_storage_buf("Particles3D Counter", 4);

        // --- Billboard quad mesh ---
        create_billboard_quad(gpu);

        // --- Cuboid mesh ---
        create_box_mesh(gpu);

        // --- Compute shader ---
        compute_shader_ = vivid::gpu::create_wgsl_shader(
            gpu->device, kParticlesCompute, "Particles3D Compute Shader");
        if (!compute_shader_) {
            std::fprintf(stderr, "[particles3d] Failed to compile compute shader\n");
            return;
        }

        // --- Bind group layout ---
        WGPUBindGroupLayoutEntry entries[5]{};

        // binding 0: particles_in (read-only storage)
        entries[0].binding = 0;
        entries[0].visibility = WGPUShaderStage_Compute;
        entries[0].buffer.type = WGPUBufferBindingType_ReadOnlyStorage;
        entries[0].buffer.minBindingSize = 0;

        // binding 1: particles_out (read-write storage)
        entries[1].binding = 1;
        entries[1].visibility = WGPUShaderStage_Compute;
        entries[1].buffer.type = WGPUBufferBindingType_Storage;
        entries[1].buffer.minBindingSize = 0;

        // binding 2: instances_out (read-write storage)
        entries[2].binding = 2;
        entries[2].visibility = WGPUShaderStage_Compute;
        entries[2].buffer.type = WGPUBufferBindingType_Storage;
        entries[2].buffer.minBindingSize = 0;

        // binding 3: params (uniform)
        entries[3].binding = 3;
        entries[3].visibility = WGPUShaderStage_Compute;
        entries[3].buffer.type = WGPUBufferBindingType_Uniform;
        entries[3].buffer.minBindingSize = sizeof(ParamsData);

        // binding 4: counter (read-write storage)
        entries[4].binding = 4;
        entries[4].visibility = WGPUShaderStage_Compute;
        entries[4].buffer.type = WGPUBufferBindingType_Storage;
        entries[4].buffer.minBindingSize = 4;

        WGPUBindGroupLayoutDescriptor bgl_desc{};
        bgl_desc.label = vivid_sv("Particles3D BGL");
        bgl_desc.entryCount = 5;
        bgl_desc.entries = entries;
        compute_bgl_ = wgpuDeviceCreateBindGroupLayout(gpu->device, &bgl_desc);

        // --- Pipeline layout ---
        WGPUPipelineLayoutDescriptor pl_desc{};
        pl_desc.label = vivid_sv("Particles3D Pipeline Layout");
        pl_desc.bindGroupLayoutCount = 1;
        pl_desc.bindGroupLayouts = &compute_bgl_;
        compute_pipe_layout_ = wgpuDeviceCreatePipelineLayout(gpu->device, &pl_desc);

        // --- Compute pipeline ---
        WGPUComputePipelineDescriptor cp_desc{};
        cp_desc.label = vivid_sv("Particles3D Compute Pipeline");
        cp_desc.layout = compute_pipe_layout_;
        cp_desc.compute.module = compute_shader_;
        cp_desc.compute.entryPoint = vivid_sv("cs_main");
        compute_pipeline_ = wgpuDeviceCreateComputePipeline(gpu->device, &cp_desc);
        if (!compute_pipeline_) {
            std::fprintf(stderr, "[particles3d] Failed to create compute pipeline\n");
            return;
        }

        // --- Bind groups (ping-pong) ---
        // Group A: reads buf_a, writes buf_b
        create_bind_group(gpu, particle_buf_a_, particle_buf_b_,
                          particle_buf_size, &bind_group_a_, "Particles3D BG A");
        // Group B: reads buf_b, writes buf_a
        create_bind_group(gpu, particle_buf_b_, particle_buf_a_,
                          particle_buf_size, &bind_group_b_, "Particles3D BG B");
    }

    void create_bind_group(const VividGpuContext* gpu,
                           WGPUBuffer read_buf, WGPUBuffer write_buf,
                           uint64_t particle_buf_size,
                           WGPUBindGroup* out_bg, const char* label) {
        uint64_t instance_buf_size = static_cast<uint64_t>(current_count_) * sizeof(vivid::gpu::InstanceData3D);
        if (instance_buf_size < 48) instance_buf_size = 48;

        WGPUBindGroupEntry entries[5]{};
        entries[0].binding = 0;
        entries[0].buffer  = read_buf;
        entries[0].offset  = 0;
        entries[0].size    = particle_buf_size;

        entries[1].binding = 1;
        entries[1].buffer  = write_buf;
        entries[1].offset  = 0;
        entries[1].size    = particle_buf_size;

        entries[2].binding = 2;
        entries[2].buffer  = instance_buf_;
        entries[2].offset  = 0;
        entries[2].size    = instance_buf_size;

        entries[3].binding = 3;
        entries[3].buffer  = params_ubo_;
        entries[3].offset  = 0;
        entries[3].size    = sizeof(ParamsData);

        entries[4].binding = 4;
        entries[4].buffer  = counter_buf_;
        entries[4].offset  = 0;
        entries[4].size    = 4;

        WGPUBindGroupDescriptor desc{};
        desc.label      = vivid_sv(label);
        desc.layout     = compute_bgl_;
        desc.entryCount = 5;
        desc.entries    = entries;
        *out_bg = wgpuDeviceCreateBindGroup(gpu->device, &desc);
    }

    void create_billboard_quad(const VividGpuContext* gpu) {
        // Unit quad in XY plane centered at origin
        vivid::gpu::Vertex3D verts[4]{};

        // Bottom-left
        verts[0].position[0] = -0.5f; verts[0].position[1] = -0.5f; verts[0].position[2] = 0.0f;
        verts[0].normal[0] = 0; verts[0].normal[1] = 0; verts[0].normal[2] = 1;
        verts[0].tangent[0] = 1; verts[0].tangent[1] = 0; verts[0].tangent[2] = 0; verts[0].tangent[3] = 1;
        verts[0].uv[0] = 0; verts[0].uv[1] = 0;

        // Bottom-right
        verts[1].position[0] = 0.5f; verts[1].position[1] = -0.5f; verts[1].position[2] = 0.0f;
        verts[1].normal[0] = 0; verts[1].normal[1] = 0; verts[1].normal[2] = 1;
        verts[1].tangent[0] = 1; verts[1].tangent[1] = 0; verts[1].tangent[2] = 0; verts[1].tangent[3] = 1;
        verts[1].uv[0] = 1; verts[1].uv[1] = 0;

        // Top-right
        verts[2].position[0] = 0.5f; verts[2].position[1] = 0.5f; verts[2].position[2] = 0.0f;
        verts[2].normal[0] = 0; verts[2].normal[1] = 0; verts[2].normal[2] = 1;
        verts[2].tangent[0] = 1; verts[2].tangent[1] = 0; verts[2].tangent[2] = 0; verts[2].tangent[3] = 1;
        verts[2].uv[0] = 1; verts[2].uv[1] = 1;

        // Top-left
        verts[3].position[0] = -0.5f; verts[3].position[1] = 0.5f; verts[3].position[2] = 0.0f;
        verts[3].normal[0] = 0; verts[3].normal[1] = 0; verts[3].normal[2] = 1;
        verts[3].tangent[0] = 1; verts[3].tangent[1] = 0; verts[3].tangent[2] = 0; verts[3].tangent[3] = 1;
        verts[3].uv[0] = 0; verts[3].uv[1] = 1;

        quad_vb_ = vivid::gpu::create_vertex_buffer(
            gpu->device, gpu->queue, verts, sizeof(verts), "Particles3D Quad VB");

        uint32_t indices[6] = { 0, 1, 2, 0, 2, 3 };
        quad_ib_ = vivid::gpu::create_index_buffer(
            gpu->device, gpu->queue, indices, 6, "Particles3D Quad IB");
    }

    void create_box_mesh(const VividGpuContext* gpu) {
        using V = vivid::gpu::Vertex3D;

        // Face data: normal, tangent, then 4 corner positions
        struct Face {
            float nx, ny, nz;
            float tx, ty, tz;
            float v[4][3];
        };
        static const Face faces[] = {
            // +Z front   tangent: +X
            { 0,0,1,  1,0,0, { {-0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f, 0.5f}, { 0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f} } },
            // -Z back    tangent: -X
            { 0,0,-1, -1,0,0, { { 0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f,-0.5f}, {-0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f,-0.5f} } },
            // +X right   tangent: -Z
            { 1,0,0,  0,0,-1, { { 0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f,-0.5f}, { 0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f, 0.5f} } },
            // -X left    tangent: +Z
            {-1,0,0,  0,0,1, { {-0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f,-0.5f} } },
            // +Y top     tangent: +X
            { 0,1,0,  1,0,0, { {-0.5f, 0.5f, 0.5f}, { 0.5f, 0.5f, 0.5f}, { 0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f,-0.5f} } },
            // -Y bottom  tangent: +X
            { 0,-1,0, 1,0,0, { {-0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f, 0.5f}, {-0.5f,-0.5f, 0.5f} } },
        };

        static const float uvs[4][2] = { {0,0}, {1,0}, {1,1}, {0,1} };

        V verts[24];
        uint32_t indices[36];
        int vi = 0, ii = 0;

        for (int f = 0; f < 6; ++f) {
            uint32_t base = static_cast<uint32_t>(vi);
            for (int i = 0; i < 4; ++i) {
                V& v = verts[vi++];
                v = {};
                v.position[0] = faces[f].v[i][0];
                v.position[1] = faces[f].v[i][1];
                v.position[2] = faces[f].v[i][2];
                v.normal[0] = faces[f].nx;
                v.normal[1] = faces[f].ny;
                v.normal[2] = faces[f].nz;
                v.tangent[0] = faces[f].tx;
                v.tangent[1] = faces[f].ty;
                v.tangent[2] = faces[f].tz;
                v.tangent[3] = 1.0f;
                v.uv[0] = uvs[i][0];
                v.uv[1] = uvs[i][1];
            }
            indices[ii++] = base + 0;
            indices[ii++] = base + 1;
            indices[ii++] = base + 2;
            indices[ii++] = base + 0;
            indices[ii++] = base + 2;
            indices[ii++] = base + 3;
        }

        box_vb_ = vivid::gpu::create_vertex_buffer(
            gpu->device, gpu->queue, verts, sizeof(verts), "Particles3D Box VB");
        box_ib_ = vivid::gpu::create_index_buffer(
            gpu->device, gpu->queue, indices, 36, "Particles3D Box IB");
    }
};

// ---------------------------------------------------------------------------
// Dedicated editor window — cluster-organized slider layout
//
// The existing inspector surfaces all 18+ params as a flat scrolling knob
// list. The editor groups them into named clusters (Emission, Physics,
// Curl Noise, Appearance, Color, Material, Learning) arranged in two
// columns and renders each numeric param as a full-width horizontal
// slider. Color gets a 4-channel swatch row. Enum params use radio
// buttons. Live parameter changes flow through commands.set_param and
// update the graph immediately.
//
// Live 3D preview is deferred — the platform doesn't expose an
// editor-owned 3D viewport yet (see candidates/particles3d.md §"Preview
// mechanism"). The cluster-organized layout is the primary UX win
// relative to the flat inspector knob grid.
// ---------------------------------------------------------------------------

VividEditorMetadata Particles3D::editor_metadata() {
    VividEditorMetadata m{};
    m.default_width  = 1100;
    m.default_height = 640;
    m.min_width      = 800;
    m.min_height     = 480;
    m.title_suffix   = "Particles3D Editor";
    return m;
}

namespace {

constexpr float kEditorInset      = 8.0f;
constexpr float kEditorTopBarH    = 32.0f;
constexpr float kEditorRowH       = 22.0f;
constexpr float kEditorClusterGap = 14.0f;
constexpr float kEditorHeaderH    = 20.0f;

// Draw a cluster header (a dim separator bar with the cluster title).
void draw_cluster_header(VividEditorContext* ctx, vivid::ui::Rect r,
                         const char* title) {
    auto& d = ctx->draw;
    void* o = d.opaque;
    const auto& th = ctx->theme;
    if (d.draw_rect) {
        d.draw_rect(o, r.x, r.y + r.h - 1.0f, r.w, 1.0f,
            {th.separator.r, th.separator.g, th.separator.b, 0.6f});
    }
    if (d.draw_text) {
        d.draw_text(o, r.x + 2.0f, r.y + 2.0f, title,
            {th.bright_text.r, th.bright_text.g,
             th.bright_text.b, 0.95f}, 0.9f);
    }
}

} // namespace

void Particles3D::draw_editor(VividEditorContext* ctx) {
    if (!ctx) return;

    namespace ui = vivid::ui;

    auto& d = ctx->draw;
    void* o = d.opaque;
    const auto& th = ctx->theme;

    auto set_named = [&](const char* name, float v) {
        if (ctx->commands.set_param)
            ctx->commands.set_param(ctx->commands.opaque, name, v);
    };

    ctx->wants_keyboard = 1;

    const float surf_w = ctx->surface_width;
    const float surf_h = ctx->surface_height;

    // Top bar.
    if (d.draw_text) {
        d.draw_text(o, kEditorInset + 4.0f, kEditorInset + 8.0f,
            "Particles3D",
            {th.bright_text.r, th.bright_text.g,
             th.bright_text.b, 0.95f}, 1.0f);
        const char* hints =
            "drag sliders to edit  ·  click radio for enums  ·  "
            "set_param flows through immediately";
        const float scale = 0.7f;
        const float hints_w = d.text_width
            ? d.text_width(o, hints, scale) : 520.0f;
        d.draw_text(o, surf_w - kEditorInset - hints_w,
            kEditorInset + 10.0f, hints,
            {th.dim_text.r, th.dim_text.g, th.dim_text.b, 0.7f}, scale);
    }

    // Body region: two columns of clusters.
    const float body_x = kEditorInset;
    const float body_y = kEditorInset + kEditorTopBarH + kEditorInset;
    const float body_w = std::max(0.0f, surf_w - 2.0f * kEditorInset);
    const float body_h = std::max(0.0f, surf_h - body_y - kEditorInset);

    vivid::draw_ui::draw_panel(d, o, body_x, body_y, body_w, body_h,
        {th.dark_bg.r, th.dark_bg.g, th.dark_bg.b, 0.92f},
        {th.separator.r, th.separator.g, th.separator.b, 0.5f}, 4.0f, 1.0f);

    auto body = ui::Rect{body_x, body_y, body_w, body_h};
    auto body_inset = ui::ui_pad(body, 12.0f);
    auto split = ui::ui_split_h(body_inset, 0.5f, 18.0f);
    auto left  = ui::LayoutCursor{split.first,  0.0f, 6.0f};
    left.cursor_x = split.first.x;
    left.cursor_y = split.first.y;
    left.remaining_w = split.first.w;
    left.remaining_h = split.first.h;
    auto right = ui::LayoutCursor{split.second, 0.0f, 6.0f};
    right.cursor_x = split.second.x;
    right.cursor_y = split.second.y;
    right.remaining_w = split.second.w;
    right.remaining_h = split.second.h;

    auto slider_int = [&](ui::LayoutCursor& c, int slot, const char* label,
                          const char* param_name, int cur, int lo, int hi) {
        auto row = ui::ui_row(c, kEditorRowH);
        auto r = ui::ui_slider_h(*ctx, row, label,
            static_cast<float>(cur),
            static_cast<float>(lo), static_cast<float>(hi),
            &editor_sliders_[slot]);
        if (r.changed) {
            const int iv = std::clamp(
                static_cast<int>(std::lround(r.value)), lo, hi);
            set_named(param_name, static_cast<float>(iv));
        }
    };
    auto slider_float = [&](ui::LayoutCursor& c, int slot, const char* label,
                            const char* param_name,
                            float cur, float lo, float hi) {
        auto row = ui::ui_row(c, kEditorRowH);
        auto r = ui::ui_slider_h(*ctx, row, label, cur, lo, hi,
            &editor_sliders_[slot]);
        if (r.changed) set_named(param_name, r.value);
    };
    auto cluster_gap = [&](ui::LayoutCursor& c) {
        ui::ui_row(c, kEditorClusterGap - 6.0f);
    };

    // ===== LEFT COLUMN =====

    draw_cluster_header(ctx, ui::ui_row(left, kEditorHeaderH), "Emission");
    slider_int  (left, 0, "Count",         "count",         count.int_value(),        1, 100000);
    slider_float(left, 1, "Rate",          "emission_rate", emission_rate.value, 0.0f, 10000.0f);
    slider_float(left, 2, "Lifetime",      "lifetime",      lifetime.value,       0.1f, 30.0f);
    cluster_gap(left);

    draw_cluster_header(ctx, ui::ui_row(left, kEditorHeaderH), "Physics");
    slider_float(left, 3, "Speed",         "speed",   speed.value,    0.0f, 20.0f);
    slider_float(left, 4, "Gravity",       "gravity", gravity.value, -20.0f, 20.0f);
    slider_float(left, 5, "Spread (deg)",  "spread",  spread.value,   0.0f, 360.0f);
    slider_float(left, 6, "Drag",          "drag",    drag.value,     0.0f, 10.0f);
    cluster_gap(left);

    draw_cluster_header(ctx, ui::ui_row(left, kEditorHeaderH), "Curl Noise");
    slider_float(left, 7,  "Strength",      "curl_strength",  curl_strength.value,  0.0f, 20.0f);
    slider_float(left, 8,  "Scale",         "noise_scale",    noise_scale.value,    0.01f, 10.0f);
    slider_float(left, 9,  "Speed",         "noise_speed",    noise_speed.value,    0.0f, 5.0f);
    slider_int  (left, 10, "Octaves",       "noise_octaves",  noise_octaves.int_value(), 1, 4);

    // ===== RIGHT COLUMN =====

    draw_cluster_header(ctx, ui::ui_row(right, kEditorHeaderH), "Appearance");
    {
        // Shape: two-option radio (Billboard / Cuboid).
        auto row = ui::ui_row(right, kEditorRowH + 4.0f);
        static const char* kShapeLabels[2] = {"Billboard", "Cuboid"};
        auto r = ui::ui_radio(*ctx, row, kShapeLabels, 2, shape.int_value());
        if (r.clicked) set_named("shape", static_cast<float>(r.value));
    }
    slider_float(right, 11, "Elongation", "elongation", elongation.value, 1.0f, 20.0f);
    slider_float(right, 12, "Size",       "size",       size.value,       0.01f, 2.0f);
    slider_float(right, 13, "Bounds",     "bounds",     bounds.value,     1.0f, 200.0f);
    cluster_gap(right);

    draw_cluster_header(ctx, ui::ui_row(right, kEditorHeaderH), "Color");
    slider_float(right, 14, "R", "r", r.value, 0.0f, 1.0f);
    slider_float(right, 15, "G", "g", g.value, 0.0f, 1.0f);
    slider_float(right, 16, "B", "b", b.value, 0.0f, 1.0f);
    slider_float(right, 17, "A", "a", a.value, 0.0f, 1.0f);
    {
        // Color swatch: filled rect showing current rgba.
        auto row = ui::ui_row(right, 22.0f);
        if (d.draw_rect) {
            // Background/transparency chequer substitute: dim track.
            d.draw_rect(o, row.x, row.y, row.w, row.h,
                {0.12f, 0.12f, 0.13f, 1.0f});
            d.draw_rect(o, row.x + 2.0f, row.y + 2.0f,
                row.w - 4.0f, row.h - 4.0f,
                {this->r.value, this->g.value, this->b.value, this->a.value});
        }
        if (d.draw_text) {
            d.draw_text(o, row.x + 8.0f, row.y + 4.0f, "Preview",
                {th.bright_text.r, th.bright_text.g,
                 th.bright_text.b, 0.8f}, 0.7f);
        }
    }
    cluster_gap(right);

    draw_cluster_header(ctx, ui::ui_row(right, kEditorHeaderH), "Material");
    slider_float(right, 18, "Emission", "emission", emission.value, 0.0f, 5.0f);
    {
        auto row = ui::ui_row(right, kEditorRowH + 4.0f);
        static const char* kUnlitLabels[2] = {"Lit", "Unlit"};
        auto rr = ui::ui_radio(*ctx, row, kUnlitLabels, 2, unlit.int_value());
        if (rr.clicked) set_named("unlit", static_cast<float>(rr.value));
    }
    cluster_gap(right);

    draw_cluster_header(ctx, ui::ui_row(right, kEditorHeaderH), "Learning mode");
    {
        auto row = ui::ui_row(right, kEditorRowH + 4.0f);
        static const char* kLearningLabels[2] = {"Advanced", "Beginner"};
        auto rr = ui::ui_radio(*ctx, row, kLearningLabels, 2, learning_mode.int_value());
        if (rr.clicked) set_named("learning_mode", static_cast<float>(rr.value));
    }
}

VIVID_REGISTER(Particles3D)
VIVID_THUMBNAIL(Particles3D)
VIVID_EDITOR(Particles3D)

VIVID_DESCRIBE_REF_TYPE(vivid::gpu::VividSceneFragment)
