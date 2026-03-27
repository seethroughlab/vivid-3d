#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_common.h"
#include <cstdio>
#include <cstring>
#include <cstdint>

// =============================================================================
// DepthMask Shader
// =============================================================================

static const char* kDepthMaskFragment = R"(
struct DepthMaskParams {
    threshold: f32,
    softness: f32,
    mode: u32,
    invert: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> params: DepthMaskParams;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var color_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
    let fs = fullscreenTriangle(vertexIndex, true);
    var out: VertexOutput;
    out.position = fs.position;
    out.uv = fs.uv;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4f {
    let dims_d = textureDimensions(depth_tex);
    let dims_c = textureDimensions(color_tex);
    let px_d = vec2i(input.uv * vec2f(f32(dims_d.x), f32(dims_d.y)));
    let px_c = input.uv;

    let depth = textureLoad(depth_tex, clamp(px_d, vec2i(0), vec2i(dims_d) - vec2i(1)), 0).r;
    let color = textureSample(color_tex, tex_sampler, px_c);

    var mask: f32 = 0.0;
    let half_soft = max(params.softness * 0.5, 0.0001);

    if (params.mode == 0u) {
        // Near: mask where depth < threshold (objects closer than threshold)
        mask = smoothstep(params.threshold + half_soft, params.threshold - half_soft, depth);
    } else if (params.mode == 1u) {
        // Far: mask where depth > threshold (objects farther than threshold)
        mask = smoothstep(params.threshold - half_soft, params.threshold + half_soft, depth);
    } else {
        // Range: mask near threshold with softness on both sides
        let dist = abs(depth - params.threshold);
        mask = 1.0 - smoothstep(0.0, half_soft, dist);
    }

    if (params.invert > 0u) {
        mask = 1.0 - mask;
    }

    return vec4f(color.rgb * mask, color.a * mask);
}
)";

// =============================================================================
// Uniform struct (must match WGSL DepthMaskParams)
// =============================================================================

struct DepthMaskUniforms {
    float threshold;
    float softness;
    uint32_t mode;
    uint32_t invert;
};

// =============================================================================
// DepthMask3D Operator
// =============================================================================

struct DepthMask3D : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "DepthMask3D";
    static constexpr bool kTimeDependent = false;

    vivid::Param<float> threshold {"threshold", 0.99f, 0.0f, 1.0f};
    vivid::Param<float> softness  {"softness",  0.1f,  0.0f, 1.0f};
    vivid::Param<int>   mode      {"mode",      0, {"Near", "Far", "Range"}};
    vivid::Param<int>   invert    {"invert",    0, {"Off", "On"}};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        out.push_back(&threshold);
        out.push_back(&softness);
        out.push_back(&mode);
        out.push_back(&invert);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"color",   VIVID_PORT_TEXTURE, VIVID_PORT_INPUT});
        out.push_back({"depth",   VIVID_PORT_TEXTURE, VIVID_PORT_INPUT});
        out.push_back({"texture", VIVID_PORT_TEXTURE, VIVID_PORT_OUTPUT});
    }

    void process_gpu(const VividGpuContext* ctx) override {
        if (!pipeline_) {
            if (!lazy_init(ctx)) {
                std::fprintf(stderr, "[depth_mask3d] lazy_init FAILED\n");
                return;
            }
        }

        // Get inputs
        WGPUTextureView color_input = nullptr;
        WGPUTextureView depth_input = nullptr;
        if (ctx->input_texture_views && ctx->input_texture_count >= 1)
            color_input = ctx->input_texture_views[0];
        if (ctx->input_texture_views && ctx->input_texture_count >= 2)
            depth_input = ctx->input_texture_views[1];

        if (!color_input || !depth_input) return;

        // Update uniforms
        DepthMaskUniforms u{};
        u.threshold = threshold.value;
        u.softness  = softness.value;
        u.mode      = static_cast<uint32_t>(mode.value);
        u.invert    = static_cast<uint32_t>(invert.value);
        wgpuQueueWriteBuffer(ctx->queue, uniform_buf_, 0, &u, sizeof(u));

        // Rebuild bind group if inputs changed
        if (color_input != cached_color_ || depth_input != cached_depth_) {
            rebuild_bind_group(ctx, color_input, depth_input);
            cached_color_ = color_input;
            cached_depth_ = depth_input;
        }

        static constexpr WGPUColor kClear{0, 0, 0, 0};

        // Single fullscreen pass → output
        vivid::gpu::run_pass(ctx->command_encoder, pipeline_, bind_group_,
                             ctx->output_texture_view, "DepthMask", kClear);
    }

    ~DepthMask3D() override {
        vivid::gpu::release(pipeline_);
        vivid::gpu::release(bgl_);
        vivid::gpu::release(pipe_layout_);
        vivid::gpu::release(shader_);
        vivid::gpu::release(uniform_buf_);
        vivid::gpu::release(sampler_);
        vivid::gpu::release(bind_group_);
    }

private:
    WGPURenderPipeline  pipeline_    = nullptr;
    WGPUBindGroupLayout bgl_         = nullptr;
    WGPUPipelineLayout  pipe_layout_ = nullptr;
    WGPUShaderModule    shader_      = nullptr;
    WGPUBuffer          uniform_buf_ = nullptr;
    WGPUSampler         sampler_     = nullptr;
    WGPUBindGroup       bind_group_  = nullptr;

    WGPUTextureView cached_color_ = nullptr;
    WGPUTextureView cached_depth_ = nullptr;

    void rebuild_bind_group(const VividGpuContext* gpu, WGPUTextureView color_input,
                            WGPUTextureView depth_input) {
        vivid::gpu::release(bind_group_);

        WGPUBindGroupEntry entries[4]{};
        entries[0].binding = 0;
        entries[0].buffer  = uniform_buf_;
        entries[0].size    = sizeof(DepthMaskUniforms);
        entries[1].binding = 1;
        entries[1].sampler = sampler_;
        entries[2].binding = 2;
        entries[2].textureView = color_input;
        entries[3].binding = 3;
        entries[3].textureView = depth_input;

        WGPUBindGroupDescriptor desc{};
        desc.label = vivid_sv("DepthMask BG");
        desc.layout = bgl_;
        desc.entryCount = 4;
        desc.entries = entries;
        bind_group_ = wgpuDeviceCreateBindGroup(gpu->device, &desc);
    }

    bool lazy_init(const VividGpuContext* gpu) {
        shader_ = vivid::gpu::create_shader(gpu->device, kDepthMaskFragment, "DepthMask Shader");
        if (!shader_) return false;

        uniform_buf_ = vivid::gpu::create_uniform_buffer(gpu->device, sizeof(DepthMaskUniforms), "DepthMask Uniforms");
        sampler_ = vivid::gpu::create_linear_sampler(gpu->device, "DepthMask Sampler");

        // --- BGL: uniform(0) + sampler(1) + color(2, float) + depth(3, unfilterable float) ---
        {
            WGPUBindGroupLayoutEntry entries[4]{};
            entries[0].binding    = 0;
            entries[0].visibility = WGPUShaderStage_Fragment;
            entries[0].buffer.type           = WGPUBufferBindingType_Uniform;
            entries[0].buffer.minBindingSize = sizeof(DepthMaskUniforms);

            entries[1].binding    = 1;
            entries[1].visibility = WGPUShaderStage_Fragment;
            entries[1].sampler.type = WGPUSamplerBindingType_Filtering;

            entries[2].binding    = 2;
            entries[2].visibility = WGPUShaderStage_Fragment;
            entries[2].texture.sampleType    = WGPUTextureSampleType_Float;
            entries[2].texture.viewDimension = WGPUTextureViewDimension_2D;

            entries[3].binding    = 3;
            entries[3].visibility = WGPUShaderStage_Fragment;
            entries[3].texture.sampleType    = WGPUTextureSampleType_UnfilterableFloat;
            entries[3].texture.viewDimension = WGPUTextureViewDimension_2D;

            WGPUBindGroupLayoutDescriptor bgl_desc{};
            bgl_desc.label = vivid_sv("DepthMask BGL");
            bgl_desc.entryCount = 4;
            bgl_desc.entries = entries;
            bgl_ = wgpuDeviceCreateBindGroupLayout(gpu->device, &bgl_desc);
        }

        // --- Pipeline layout ---
        {
            WGPUPipelineLayoutDescriptor pl{};
            pl.label = vivid_sv("DepthMask PL");
            pl.bindGroupLayoutCount = 1;
            pl.bindGroupLayouts = &bgl_;
            pipe_layout_ = wgpuDeviceCreatePipelineLayout(gpu->device, &pl);
        }

        // --- Pipeline → output format ---
        pipeline_ = vivid::gpu::create_pipeline(gpu->device, shader_,
                                                 pipe_layout_,
                                                 gpu->output_format,
                                                 "DepthMask Pipeline");

        if (!pipeline_) return false;

        return true;
    }
};

VIVID_REGISTER(DepthMask3D)
