#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_common.h"
#include "operator_api/type_id.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

// =============================================================================
// InstancedRender — renders N instances of an input mesh to a texture.
//
// Input ports:
//   "mesh"       VIVID_PORT_HANDLE     (required) — geometry to draw
//   "transforms" VIVID_PORT_HANDLE  (optional) — per-instance vec4 transforms
//   "positions"  VIVID_PORT_SPREAD (fallback) — [x0,y0, x1,y1, ...] pairs
//
// Output: "texture" VIVID_PORT_TEXTURE
//
// Per-instance data (vec4f): xyz = translation, w = unused.
// Vertex shader reads @location(0) vec3f position from mesh vertex buffer,
// scales by the `scale` param, then translates by the per-instance vec4.xyz.
//
// v1 constraint: mesh must have shader_location=0 = Float32x3 position
//                at offset 0 in the vertex buffer.
// =============================================================================

// WGSL shader — custom vertex + simple white fragment.
// Does NOT use FULLSCREEN_VERTEX_WGSL; compiled manually.
static const char* kInstancedRenderShader = R"(

struct Uniforms {
    scale: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

struct VertexOutput {
    @builtin(position) position: vec4f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(1) @binding(0) var<storage, read> instance_transforms: array<vec4f>;

@vertex
fn vs_main(@location(0) pos: vec3f,
           @builtin(instance_index) ii: u32) -> VertexOutput {
    let t = instance_transforms[ii];
    var out: VertexOutput;
    out.position = vec4f(pos * uniforms.scale + t.xyz, 1.0);
    return out;
}

@fragment
fn fs_main() -> @location(0) vec4f {
    return vec4f(1.0, 1.0, 1.0, 1.0);
}
)";

// Uniforms struct (16 bytes, vec4-aligned)
struct InstancedRenderUniforms {
    float scale;
    float _pad0, _pad1, _pad2;
};

// =============================================================================
// InstancedRender Operator
// =============================================================================

struct InstancedRender : vivid::GpuOperatorBase {
    static constexpr const char* kName   = "InstancedRender";
    static constexpr bool kTimeDependent = false;

    vivid::Param<float> scale {"scale", 0.1f, 0.001f, 1.0f};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        out.push_back(&scale);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(VIVID_HANDLE_PORT("mesh",       VIVID_PORT_INPUT,  VividMesh));
        out.push_back(VIVID_HANDLE_PORT("transforms", VIVID_PORT_INPUT,  VividComputeBuffer));
        out.push_back({"positions",  VIVID_PORT_SPREAD, VIVID_PORT_INPUT});
        out.push_back({"texture",    VIVID_PORT_TEXTURE,    VIVID_PORT_OUTPUT});
    }

    void process_gpu(const VividGpuContext* ctx) override {
        if (!ctx->output_texture_view) return;

        // --- Require input mesh ---
        VividMesh* mesh = nullptr;
        if (ctx->input_handle_count > 0 && ctx->input_handles && ctx->input_handles[0])
            mesh = static_cast<VividMesh*>(ctx->input_handles[0]);
        if (!mesh || !mesh->vertex_buffer || mesh->vertex_count == 0) {
            // No mesh: clear to transparent and return
            clear_output(ctx);
            return;
        }

        // --- Rebuild render pipeline if mesh stride changed ---
        if (!pipeline_ || mesh->vertex_stride != built_stride_) {
            if (!lazy_init(ctx, mesh->vertex_stride)) {
                std::fprintf(stderr, "[instanced_render] lazy_init FAILED\n");
                return;
            }
        }

        // --- Determine instance transforms source ---
        WGPUBuffer  transform_buf      = nullptr;
        uint64_t    transform_buf_size = 0;
        uint32_t    instance_count     = 0;

        if (ctx->input_handle_count > 1 && ctx->input_handles && ctx->input_handles[1]) {
            // Compute path: use the compute buffer directly
            VividComputeBuffer* cb = static_cast<VividComputeBuffer*>(ctx->input_handles[1]);
            transform_buf      = cb->buffer;
            transform_buf_size = cb->size_bytes;
            instance_count     = cb->element_count;
        } else {
            // Positions spread fallback: [x0,y0, x1,y1, ...]
            uint32_t    spread_len  = 0;
            const float* spread_data = nullptr;
            if (ctx->input_spreads && ctx->input_spreads[2].length > 0) {
                spread_len  = ctx->input_spreads[2].length;
                spread_data = ctx->input_spreads[2].data;
            }
            instance_count = spread_len / 2;
            if (instance_count == 0) instance_count = 1;  // one instance at origin

            // Rebuild local storage buffer when count changes
            if (instance_count != storage_count_) {
                rebuild_storage(ctx, instance_count);
            }

            // Upload vec4 translations from positions spread
            if (storage_buf_ && instance_count > 0) {
                std::vector<float> transforms(instance_count * 4, 0.0f);
                for (uint32_t i = 0; i < instance_count; ++i) {
                    if (spread_data && (i * 2 + 1) < spread_len) {
                        transforms[i * 4 + 0] = spread_data[i * 2 + 0];  // x
                        transforms[i * 4 + 1] = spread_data[i * 2 + 1];  // y
                    }
                    // z and w remain 0
                }
                uint64_t bytes = static_cast<uint64_t>(instance_count) * 4 * sizeof(float);
                wgpuQueueWriteBuffer(ctx->queue, storage_buf_, 0, transforms.data(), bytes);
            }

            transform_buf      = storage_buf_;
            transform_buf_size = static_cast<uint64_t>(instance_count) * 16;
        }

        if (!transform_buf || instance_count == 0) {
            clear_output(ctx);
            return;
        }

        // --- Rebuild bind group 1 when transform buffer changes ---
        if (transform_buf != cached_transform_buf_ ||
            transform_buf_size != cached_transform_buf_size_) {
            rebuild_transform_bind_group(ctx, transform_buf, transform_buf_size);
        }

        // --- Upload uniforms ---
        InstancedRenderUniforms u{};
        u.scale = scale.value;
        wgpuQueueWriteBuffer(ctx->queue, uniform_buf_, 0, &u, sizeof(u));

        // --- Render pass ---
        WGPURenderPassColorAttachment color_att{};
        color_att.view         = ctx->output_texture_view;
        color_att.depthSlice   = WGPU_DEPTH_SLICE_UNDEFINED;
        color_att.loadOp       = WGPULoadOp_Clear;
        color_att.storeOp      = WGPUStoreOp_Store;
        color_att.clearValue   = {0.0, 0.0, 0.0, 0.0};

        WGPURenderPassDescriptor rp_desc{};
        rp_desc.label                = vivid_sv("InstancedRender Pass");
        rp_desc.colorAttachmentCount = 1;
        rp_desc.colorAttachments     = &color_att;

        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
            ctx->command_encoder, &rp_desc);

        wgpuRenderPassEncoderSetPipeline(pass, pipeline_);
        wgpuRenderPassEncoderSetBindGroup(pass, 0, bind_group0_, 0, nullptr);
        if (transform_bind_group_)
            wgpuRenderPassEncoderSetBindGroup(pass, 1, transform_bind_group_, 0, nullptr);

        // Bind mesh vertex buffer (slot 0)
        wgpuRenderPassEncoderSetVertexBuffer(pass, 0, mesh->vertex_buffer,
                                             mesh->vertex_buffer_offset,
                                             WGPU_WHOLE_SIZE);

        if (mesh->index_buffer && mesh->index_count > 0) {
            wgpuRenderPassEncoderSetIndexBuffer(pass, mesh->index_buffer,
                                                mesh->index_format, 0, WGPU_WHOLE_SIZE);
            wgpuRenderPassEncoderDrawIndexed(pass, mesh->index_count,
                                             instance_count, 0, 0, 0);
        } else {
            wgpuRenderPassEncoderDraw(pass, mesh->vertex_count, instance_count, 0, 0);
        }

        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }

    ~InstancedRender() override {
        vivid::gpu::release(pipeline_);
        vivid::gpu::release(bind_group0_);
        vivid::gpu::release(transform_bind_group_);
        vivid::gpu::release(bind_layout0_);
        vivid::gpu::release(bind_layout1_);
        vivid::gpu::release(uniform_buf_);
        vivid::gpu::release(storage_buf_);
        vivid::gpu::release(shader_);
        vivid::gpu::release(pipe_layout_);
    }

private:
    WGPURenderPipeline  pipeline_              = nullptr;
    WGPUBindGroup       bind_group0_           = nullptr;  // uniforms
    WGPUBindGroup       transform_bind_group_  = nullptr;  // instance transforms
    WGPUBindGroupLayout bind_layout0_          = nullptr;
    WGPUBindGroupLayout bind_layout1_          = nullptr;
    WGPUBuffer          uniform_buf_           = nullptr;
    WGPUBuffer          storage_buf_           = nullptr;  // local fallback buffer
    WGPUShaderModule    shader_                = nullptr;
    WGPUPipelineLayout  pipe_layout_           = nullptr;
    uint32_t            built_stride_          = 0;
    uint32_t            storage_count_         = 0;
    WGPUBuffer          cached_transform_buf_  = nullptr;
    uint64_t            cached_transform_buf_size_ = 0;

    // Clear the output texture to transparent
    void clear_output(const VividGpuContext* ctx) {
        WGPURenderPassColorAttachment att{};
        att.view       = ctx->output_texture_view;
        att.depthSlice = WGPU_DEPTH_SLICE_UNDEFINED;
        att.loadOp     = WGPULoadOp_Clear;
        att.storeOp    = WGPUStoreOp_Store;
        att.clearValue = {0.0, 0.0, 0.0, 0.0};
        WGPURenderPassDescriptor rp{};
        rp.label                = vivid_sv("InstancedRender Clear");
        rp.colorAttachmentCount = 1;
        rp.colorAttachments     = &att;
        WGPURenderPassEncoder pass = wgpuCommandEncoderBeginRenderPass(
            ctx->command_encoder, &rp);
        wgpuRenderPassEncoderEnd(pass);
        wgpuRenderPassEncoderRelease(pass);
    }

    void rebuild_storage(const VividGpuContext* ctx, uint32_t count) {
        vivid::gpu::release(storage_buf_);
        vivid::gpu::release(transform_bind_group_);
        cached_transform_buf_      = nullptr;
        cached_transform_buf_size_ = 0;
        storage_count_ = count;
        if (count == 0) return;

        uint64_t bytes = static_cast<uint64_t>(count) * 4 * sizeof(float);
        if (bytes < 16) bytes = 16;

        WGPUBufferDescriptor bd{};
        bd.label = vivid_sv("InstancedRender Storage");
        bd.size  = bytes;
        bd.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        storage_buf_ = wgpuDeviceCreateBuffer(ctx->device, &bd);
    }

    void rebuild_transform_bind_group(const VividGpuContext* ctx,
                                      WGPUBuffer buf, uint64_t size) {
        vivid::gpu::release(transform_bind_group_);
        cached_transform_buf_      = buf;
        cached_transform_buf_size_ = size;
        if (!buf || !bind_layout1_) return;

        WGPUBindGroupEntry entry{};
        entry.binding = 0;
        entry.buffer  = buf;
        entry.offset  = 0;
        entry.size    = size;

        WGPUBindGroupDescriptor bg_desc{};
        bg_desc.label      = vivid_sv("InstancedRender Transform BG");
        bg_desc.layout     = bind_layout1_;
        bg_desc.entryCount = 1;
        bg_desc.entries    = &entry;
        transform_bind_group_ = wgpuDeviceCreateBindGroup(ctx->device, &bg_desc);
    }

    bool lazy_init(const VividGpuContext* ctx, uint32_t vertex_stride) {
        // Release old pipeline resources when rebuilding for a new mesh stride
        vivid::gpu::release(pipeline_);
        vivid::gpu::release(bind_group0_);
        vivid::gpu::release(transform_bind_group_);
        vivid::gpu::release(bind_layout0_);
        vivid::gpu::release(bind_layout1_);
        vivid::gpu::release(pipe_layout_);
        cached_transform_buf_      = nullptr;
        cached_transform_buf_size_ = 0;
        built_stride_ = vertex_stride;

        // Compile shader (no fullscreen vertex preamble — we have our own VS)
        std::string wgsl = std::string(vivid::gpu::WGSL_CONSTANTS) + kInstancedRenderShader;
        WGPUShaderSourceWGSL wgsl_src{};
        wgsl_src.chain.sType = WGPUSType_ShaderSourceWGSL;
        wgsl_src.code = vivid_sv(wgsl.c_str());

        WGPUShaderModuleDescriptor sm_desc{};
        sm_desc.nextInChain = &wgsl_src.chain;
        sm_desc.label = vivid_sv("InstancedRender Shader");

        // Release previous shader if rebuilding
        vivid::gpu::release(shader_);
        shader_ = wgpuDeviceCreateShaderModule(ctx->device, &sm_desc);
        if (!shader_) return false;

        // Create uniform buffer once (kept across stride rebuilds)
        if (!uniform_buf_) {
            uniform_buf_ = vivid::gpu::create_uniform_buffer(
                ctx->device, sizeof(InstancedRenderUniforms), "InstancedRender Uniforms");
        }

        // --- Bind group layout 0: uniform at binding 0 ---
        WGPUBindGroupLayoutEntry bgl0_entry{};
        bgl0_entry.binding    = 0;
        bgl0_entry.visibility = WGPUShaderStage_Vertex;
        bgl0_entry.buffer.type            = WGPUBufferBindingType_Uniform;
        bgl0_entry.buffer.minBindingSize  = sizeof(InstancedRenderUniforms);

        WGPUBindGroupLayoutDescriptor bgl0_desc{};
        bgl0_desc.label      = vivid_sv("InstancedRender BGL0");
        bgl0_desc.entryCount = 1;
        bgl0_desc.entries    = &bgl0_entry;
        bind_layout0_ = wgpuDeviceCreateBindGroupLayout(ctx->device, &bgl0_desc);

        // Bind group 0: uniforms
        WGPUBindGroupEntry bg0_entry{};
        bg0_entry.binding = 0;
        bg0_entry.buffer  = uniform_buf_;
        bg0_entry.offset  = 0;
        bg0_entry.size    = sizeof(InstancedRenderUniforms);

        WGPUBindGroupDescriptor bg0_desc{};
        bg0_desc.label      = vivid_sv("InstancedRender BG0");
        bg0_desc.layout     = bind_layout0_;
        bg0_desc.entryCount = 1;
        bg0_desc.entries    = &bg0_entry;
        bind_group0_ = wgpuDeviceCreateBindGroup(ctx->device, &bg0_desc);

        // --- Bind group layout 1: storage buffer at binding 0 ---
        WGPUBindGroupLayoutEntry bgl1_entry{};
        bgl1_entry.binding    = 0;
        bgl1_entry.visibility = WGPUShaderStage_Vertex;
        bgl1_entry.buffer.type           = WGPUBufferBindingType_ReadOnlyStorage;
        bgl1_entry.buffer.minBindingSize = 0;

        WGPUBindGroupLayoutDescriptor bgl1_desc{};
        bgl1_desc.label      = vivid_sv("InstancedRender BGL1");
        bgl1_desc.entryCount = 1;
        bgl1_desc.entries    = &bgl1_entry;
        bind_layout1_ = wgpuDeviceCreateBindGroupLayout(ctx->device, &bgl1_desc);

        // --- Pipeline layout ---
        WGPUBindGroupLayout layouts[2] = {bind_layout0_, bind_layout1_};
        WGPUPipelineLayoutDescriptor pl_desc{};
        pl_desc.label                = vivid_sv("InstancedRender Pipeline Layout");
        pl_desc.bindGroupLayoutCount = 2;
        pl_desc.bindGroupLayouts     = layouts;
        pipe_layout_ = wgpuDeviceCreatePipelineLayout(ctx->device, &pl_desc);

        // --- Vertex buffer layout: stride from mesh, pos at location 0 ---
        WGPUVertexAttribute vattr{};
        vattr.format         = WGPUVertexFormat_Float32x3;
        vattr.offset         = 0;
        vattr.shaderLocation = 0;

        WGPUVertexBufferLayout vb_layout{};
        vb_layout.arrayStride    = vertex_stride;
        vb_layout.stepMode       = WGPUVertexStepMode_Vertex;
        vb_layout.attributeCount = 1;
        vb_layout.attributes     = &vattr;

        // --- Alpha-blended render pipeline ---
        WGPUBlendState blend{};
        blend.color.srcFactor = WGPUBlendFactor_One;
        blend.color.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blend.color.operation = WGPUBlendOperation_Add;
        blend.alpha.srcFactor = WGPUBlendFactor_One;
        blend.alpha.dstFactor = WGPUBlendFactor_OneMinusSrcAlpha;
        blend.alpha.operation = WGPUBlendOperation_Add;

        WGPUColorTargetState color_target{};
        color_target.format    = ctx->output_format;
        color_target.blend     = &blend;
        color_target.writeMask = WGPUColorWriteMask_All;

        WGPUFragmentState fragment{};
        fragment.module      = shader_;
        fragment.entryPoint  = vivid_sv("fs_main");
        fragment.targetCount = 1;
        fragment.targets     = &color_target;

        WGPURenderPipelineDescriptor rp_desc{};
        rp_desc.label                    = vivid_sv("InstancedRender Pipeline");
        rp_desc.layout                   = pipe_layout_;
        rp_desc.vertex.module            = shader_;
        rp_desc.vertex.entryPoint        = vivid_sv("vs_main");
        rp_desc.vertex.bufferCount       = 1;
        rp_desc.vertex.buffers           = &vb_layout;
        rp_desc.primitive.topology       = WGPUPrimitiveTopology_TriangleList;
        rp_desc.primitive.frontFace      = WGPUFrontFace_CCW;
        rp_desc.primitive.cullMode       = WGPUCullMode_None;
        rp_desc.multisample.count        = 1;
        rp_desc.multisample.mask         = 0xFFFFFFFF;
        rp_desc.fragment                 = &fragment;

        pipeline_ = wgpuDeviceCreateRenderPipeline(ctx->device, &rp_desc);
        return pipeline_ != nullptr;
    }
};

VIVID_REGISTER(InstancedRender)
