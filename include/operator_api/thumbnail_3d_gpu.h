#pragma once

#include "operator_api/thumbnail.h"
#include "operator_api/gpu_3d.h"
#include "linmath.h"
#include <algorithm>
#include <cmath>

namespace vivid::thumb3d_gpu {

struct ThumbCamera {
    float eye[3];
    float target[3];
    float up[3];
    float fov_y;
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
    cam.fov_y = 0.6f;
    cam.aspect = static_cast<float>(w) / static_cast<float>(h);
    cam.near_z = dist * 0.01f;
    cam.far_z = dist * 4.0f;
    return cam;
}

inline void compute_aabb(const float* vert_data, uint32_t vert_count,
                         uint32_t vert_stride, uint32_t pos_offset,
                         float bmin[3], float bmax[3]) {
    if (!vert_data || vert_count == 0) {
        bmin[0] = bmin[1] = bmin[2] = -0.5f;
        bmax[0] = bmax[1] = bmax[2] = 0.5f;
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
            bmin[j] = std::min(bmin[j], p[j]);
            bmax[j] = std::max(bmax[j], p[j]);
        }
    }
}

struct ThumbnailUniforms {
    float mvp[16];
    float center[4];
    float extent[4];
    float color[4];
};

inline void render_mesh(const VividThumbnailContext* ctx,
                        WGPUBuffer vertex_buffer,
                        uint64_t vertex_buf_size,
                        WGPUBuffer index_buffer,
                        uint32_t index_count,
                        uint32_t vertex_stride,
                        WGPUPrimitiveTopology topology,
                        const float bmin[3],
                        const float bmax[3],
                        const float color[3] = nullptr) {
    if (!ctx || !ctx->device || !ctx->queue || !ctx->command_encoder ||
        !ctx->thumbnail_texture_view || !vertex_buffer || !index_buffer ||
        vertex_buf_size == 0 || index_count == 0) {
        return;
    }

    static const char* kShader = R"(
struct Uniforms {
    mvp: mat4x4f,
    center: vec4f,
    extent: vec4f,
    color: vec4f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

struct VertexIn {
    @location(0) position: vec3f,
};

struct VertexOut {
    @builtin(position) position: vec4f,
    @location(0) world: vec3f,
};

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.world = input.position;
    out.position = uniforms.mvp * vec4f(input.position, 1.0);
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4f {
    let safe_extent = max(uniforms.extent.xyz, vec3f(0.001, 0.001, 0.001));
    let pseudo_normal = normalize((input.world - uniforms.center.xyz) / safe_extent);
    let light_dir = normalize(vec3f(0.45, 0.72, 0.55));
    let shade = 0.24 + 0.76 * max(dot(pseudo_normal, light_dir), 0.0);
    return vec4f(uniforms.color.rgb * shade, uniforms.color.a);
}
)";

    WGPUShaderModule shader = vivid::thumbnail::create_shader(ctx->device, kShader, "3D Thumb Shader");
    WGPUBuffer uniform_buf = vivid::thumbnail::create_uniform_buffer(ctx->device, sizeof(ThumbnailUniforms), "3D Thumb Uniforms");
    WGPUBindGroupLayout bind_layout = vivid::thumbnail::create_uniform_bind_layout(ctx->device, sizeof(ThumbnailUniforms), "3D Thumb BGL");
    WGPUPipelineLayout pipe_layout = vivid::thumbnail::create_pipeline_layout(ctx->device, bind_layout, "3D Thumb Pipeline Layout");
    WGPUBindGroup bind_group = vivid::thumbnail::create_uniform_bind_group(ctx->device, bind_layout, uniform_buf, sizeof(ThumbnailUniforms), "3D Thumb BG");

    WGPUVertexAttribute attr{};
    attr.format = WGPUVertexFormat_Float32x3;
    attr.offset = 0;
    attr.shaderLocation = 0;

    WGPUVertexBufferLayout layout{};
    layout.arrayStride = vertex_stride;
    layout.stepMode = WGPUVertexStepMode_Vertex;
    layout.attributeCount = 1;
    layout.attributes = &attr;

    vivid::gpu::Pipeline3DDesc pipe_desc{};
    pipe_desc.shader = shader;
    pipe_desc.layout = pipe_layout;
    pipe_desc.color_format = ctx->thumbnail_format;
    pipe_desc.vertex_layouts = &layout;
    pipe_desc.vertex_layout_count = 1;
    pipe_desc.cull_mode = WGPUCullMode_Back;
    pipe_desc.front_face = WGPUFrontFace_CCW;
    pipe_desc.topology = topology;
    pipe_desc.label = "3D Thumb Pipeline";
    WGPURenderPipeline pipeline = vivid::gpu::create_3d_pipeline(ctx->device, pipe_desc);

    WGPUTexture depth_tex = vivid::gpu::create_depth_texture(
        ctx->device, ctx->thumbnail_width, ctx->thumbnail_height, "3D Thumb Depth");
    WGPUTextureView depth_view = vivid::gpu::create_depth_view(depth_tex, "3D Thumb Depth View");

    ThumbCamera cam = camera_from_bounds(bmin, bmax, ctx->thumbnail_width, ctx->thumbnail_height);
    vec3 eye = { cam.eye[0], cam.eye[1], cam.eye[2] };
    vec3 target = { cam.target[0], cam.target[1], cam.target[2] };
    vec3 up = { cam.up[0], cam.up[1], cam.up[2] };
    mat4x4 view, proj, mvp;
    mat4x4_look_at(view, eye, target, up);
    mat4x4_perspective(proj, cam.fov_y, cam.aspect, cam.near_z, cam.far_z);
    mat4x4_mul(mvp, proj, view);

    ThumbnailUniforms uniforms{};
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            uniforms.mvp[r * 4 + c] = mvp[r][c];
        }
    }
    uniforms.center[0] = (bmin[0] + bmax[0]) * 0.5f;
    uniforms.center[1] = (bmin[1] + bmax[1]) * 0.5f;
    uniforms.center[2] = (bmin[2] + bmax[2]) * 0.5f;
    uniforms.center[3] = 1.0f;
    uniforms.extent[0] = std::max(0.001f, (bmax[0] - bmin[0]) * 0.5f);
    uniforms.extent[1] = std::max(0.001f, (bmax[1] - bmin[1]) * 0.5f);
    uniforms.extent[2] = std::max(0.001f, (bmax[2] - bmin[2]) * 0.5f);
    uniforms.extent[3] = 1.0f;
    uniforms.color[0] = color ? color[0] : 0.65f;
    uniforms.color[1] = color ? color[1] : 0.72f;
    uniforms.color[2] = color ? color[2] : 0.80f;
    uniforms.color[3] = 0.95f;
    wgpuQueueWriteBuffer(ctx->queue, uniform_buf, 0, &uniforms, sizeof(uniforms));

    WGPURenderPassEncoder pass = vivid::gpu::begin_3d_pass(
        ctx->command_encoder,
        ctx->thumbnail_texture_view,
        depth_view,
        "3D Thumbnail Pass",
        WGPUColor{18.0 / 255.0, 20.0 / 255.0, 23.0 / 255.0, 230.0 / 255.0});
    wgpuRenderPassEncoderSetPipeline(pass, pipeline);
    wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group, 0, nullptr);
    wgpuRenderPassEncoderSetVertexBuffer(pass, 0, vertex_buffer, 0, vertex_buf_size);
    wgpuRenderPassEncoderSetIndexBuffer(pass, index_buffer, WGPUIndexFormat_Uint32, 0, static_cast<uint64_t>(index_count) * sizeof(uint32_t));
    wgpuRenderPassEncoderDrawIndexed(pass, index_count, 1, 0, 0, 0);
    wgpuRenderPassEncoderEnd(pass);
    wgpuRenderPassEncoderRelease(pass);

    vivid::gpu::release(depth_view);
    vivid::gpu::release(depth_tex);
    vivid::gpu::release(pipeline);
    vivid::gpu::release(bind_group);
    vivid::gpu::release(pipe_layout);
    vivid::gpu::release(bind_layout);
    vivid::gpu::release(uniform_buf);
    vivid::gpu::release(shader);
}

} // namespace vivid::thumb3d_gpu
