#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include "operator_api/thumbnail.h"
#include "operator_api/draw_plot_helpers.h"
#include <algorithm>
#include <cstdio>
#include <cstring>

// =============================================================================
// Material3D — PBR material wrapper (scene-in + 4 texture inputs → scene-out)
// =============================================================================

/**
 * @brief Applies surface shading parameters to incoming 3D geometry.
 *
 * Material3D centralizes base color, metallic, roughness, emission, and toon controls so look
 * development can stay separate from geometry generation.
 *
 * @param color_r Red channel of the base color.
 * @param roughness Surface roughness.
 * @param metallic Metallic response.
 * @param emission Self-illumination amount.
 * @param toon_levels Number of toon-shading steps.
 */
struct Material3D : vivid::OperatorBase, vivid::GpuProcessable {
    static constexpr const char* kName   = "Material3D";
    static constexpr bool kTimeDependent = false;

    vivid::Param<float> color_r    {"color_r",    1.0f, 0.0f, 1.0f};
    vivid::Param<float> color_g    {"color_g",    1.0f, 0.0f, 1.0f};
    vivid::Param<float> color_b    {"color_b",    1.0f, 0.0f, 1.0f};
    vivid::Param<float> color_a    {"color_a",    1.0f, 0.0f, 1.0f};
    vivid::Param<float> roughness  {"roughness",  0.5f, 0.0f, 1.0f};
    vivid::Param<float> metallic   {"metallic",   0.0f, 0.0f, 1.0f};
    vivid::Param<float> emission   {"emission",   0.0f, 0.0f, 10.0f};
    vivid::Param<float> unlit      {"unlit",      0.0f, 0.0f, 1.0f};
    vivid::Param<float> shading    {"shading",    0.0f, 0.0f, 1.0f};
    vivid::Param<float> toon_levels{"toon_levels",4.0f, 1.0f, 16.0f};

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        vivid::param_group(color_r, "Material");
        vivid::param_group(color_g, "Material");
        vivid::param_group(color_b, "Material");
        vivid::param_group(color_a, "Material");
        vivid::display_hint(color_r, VIVID_DISPLAY_COLOR);
        vivid::display_hint(color_g, VIVID_DISPLAY_COLOR);
        vivid::display_hint(color_b, VIVID_DISPLAY_COLOR);
        vivid::param_group(roughness, "Material");
        vivid::param_group(metallic, "Material");
        vivid::param_group(emission, "Material");
        vivid::param_group(unlit, "Material");
        vivid::param_group(shading, "Material");
        vivid::param_group(toon_levels, "Material");

        out.push_back(&color_r);
        out.push_back(&color_g);
        out.push_back(&color_b);
        out.push_back(&color_a);
        out.push_back(&roughness);
        out.push_back(&metallic);
        out.push_back(&emission);
        out.push_back(&unlit);
        out.push_back(&shading);
        out.push_back(&toon_levels);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_INPUT));
        out.push_back({"albedo_map",             VIVID_PORT_TEXTURE, VIVID_PORT_INPUT});
        out.push_back({"normal_map",             VIVID_PORT_TEXTURE, VIVID_PORT_INPUT});
        out.push_back({"roughness_metallic_map", VIVID_PORT_TEXTURE, VIVID_PORT_INPUT});
        out.push_back({"emission_map",           VIVID_PORT_TEXTURE, VIVID_PORT_INPUT});
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

        // Label: shading mode. Unlit wins, then toon/PBR from `shading` blend.
        const char* label;
        if (unlit.value > 0.5f)           label = "Unlit";
        else if (shading.value > 0.5f)    label = "Toon";
        else                              label = "PBR";
        vivid::draw_plot::draw_thumb_label(d, o, 6.0f, 3.0f, label);

        // Badge: "R.50 M.00" — roughness + metallic at a glance.
        char badge[24];
        std::snprintf(badge, sizeof(badge), "R%.2f M%.2f", roughness.value, metallic.value);
        float bw = d.text_width ? d.text_width(o, badge, 0.8f) : 72.0f;
        vivid::draw_plot::draw_thumb_value(d, o, w - bw - 6.0f, 3.0f, bw, badge);

        // Body: color swatch with a vertical gradient that gets tighter as roughness drops
        // (sharp highlight band for glossy, diffuse wash for matte).
        float body_x = 10.0f;
        float body_y = 18.0f;
        float body_w = w - 20.0f;
        float body_h = h - body_y - 6.0f;

        float cr = color_r.value, cg = color_g.value, cb = color_b.value;
        float alpha = std::clamp(color_a.value, 0.2f, 1.0f);
        d.draw_rounded_rect(o, body_x, body_y, body_w, body_h, 4.0f,
                            VividColor{cr, cg, cb, alpha});

        // Highlight strip: narrows + brightens as roughness decreases.
        // Pull toward white for dielectrics, toward the base color for metals.
        float smooth = 1.0f - std::clamp(roughness.value, 0.0f, 1.0f);
        float strip_h = std::max(3.0f, body_h * (0.1f + 0.6f * smooth));
        float strip_y = body_y + body_h * 0.15f;
        float lift = 0.35f + 0.65f * smooth;
        float hr = cr + (1.0f - cr) * lift * (1.0f - metallic.value);
        float hg = cg + (1.0f - cg) * lift * (1.0f - metallic.value);
        float hb = cb + (1.0f - cb) * lift * (1.0f - metallic.value);
        d.draw_rounded_rect(o, body_x + 4.0f, strip_y, body_w - 8.0f, strip_h, 2.0f,
                            VividColor{hr, hg, hb, 0.55f + 0.35f * smooth});

        // Metallic indicator: right-edge band when metallic > 0.
        if (metallic.value > 0.05f) {
            float band_w = 3.0f;
            d.draw_rounded_rect(o, body_x + body_w - band_w - 2.0f, body_y + 3.0f,
                                band_w, body_h - 6.0f, 1.5f,
                                VividColor{0.95f, 0.93f, 0.85f, 0.6f + 0.4f * metallic.value});
        }

        // Emission halo: outer glow when emission > 0.
        if (emission.value > 0.05f) {
            float e = std::clamp(emission.value / 2.0f, 0.0f, 1.0f);
            d.draw_rounded_rect(o, body_x - 1.5f, body_y - 1.5f,
                                body_w + 3.0f, body_h + 3.0f, 5.0f,
                                VividColor{cr, cg, cb, 0.25f * e});
        }
    }

    void process_gpu(const VividGpuContext* ctx) override {
        // Need scene input
        bool has_input = ctx->custom_input_count > 0 &&
                         vivid::gpu::scene_input(ctx, 0) != nullptr;
        if (!has_input) return;

        if (!inited_) {
            if (!lazy_init(ctx)) return;
        }

        // Resolve 4 texture input views (fallback if disconnected)
        WGPUTextureView views[4];
        for (int i = 0; i < 4; ++i) {
            uint32_t tex_idx = static_cast<uint32_t>(i);
            if (tex_idx < ctx->input_texture_count &&
                ctx->input_texture_views && ctx->input_texture_views[tex_idx]) {
                views[i] = ctx->input_texture_views[tex_idx];
            } else {
                views[i] = fallback_views_[i];
            }
        }

        // Rebuild bind group if texture views changed
        bool views_changed = false;
        for (int i = 0; i < 4; ++i) {
            if (views[i] != cached_views_[i]) { views_changed = true; break; }
        }
        if (views_changed || !tex_bind_group_) {
            vivid::gpu::release(tex_bind_group_);

            WGPUBindGroupEntry entries[5]{};
            entries[0].binding = 0;
            entries[0].sampler = sampler_;
            for (int i = 0; i < 4; ++i) {
                entries[i + 1].binding = static_cast<uint32_t>(i + 1);
                entries[i + 1].textureView = views[i];
                cached_views_[i] = views[i];
            }

            WGPUBindGroupDescriptor desc{};
            desc.label = vivid_sv("Material3D Tex BG");
            desc.layout = tex_bind_layout_;
            desc.entryCount = 5;
            desc.entries = entries;
            tex_bind_group_ = wgpuDeviceCreateBindGroup(ctx->device, &desc);
        }

        // Build output fragment with material properties
        vivid::gpu::scene_fragment_identity(output_);
        output_.color[0] = color_r.value;
        output_.color[1] = color_g.value;
        output_.color[2] = color_b.value;
        output_.color[3] = color_a.value;
        output_.roughness    = roughness.value;
        output_.metallic     = metallic.value;
        output_.emission     = emission.value;
        output_.unlit        = unlit.value > 0.5f;
        output_.shading_mode = shading.value;
        output_.toon_levels  = toon_levels.value;
        output_.pipeline_flags = vivid::gpu::kPipelineTextured;
        output_.material_texture_binds = tex_bind_group_;

        // No geometry on wrapper — just wraps input as child
        output_.vertex_buffer   = nullptr;
        output_.vertex_buf_size = 0;
        output_.index_buffer    = nullptr;
        output_.index_count     = 0;
        output_.pipeline        = nullptr;
        output_.material_binds  = nullptr;
        output_.fragment_type   = vivid::gpu::VividSceneFragment::GEOMETRY;

        child_ = vivid::gpu::scene_input(ctx, 0);
        output_.children    = &child_;
        output_.child_count = 1;

        ctx->custom_outputs[0] = &output_;
    }

    ~Material3D() override {
        vivid::gpu::release(tex_bind_group_);
        vivid::gpu::release(tex_bind_layout_);
        vivid::gpu::release(sampler_);
        for (int i = 0; i < 4; ++i) {
            vivid::gpu::release(fallback_views_[i]);
            vivid::gpu::release(fallback_textures_[i]);
        }
    }

private:
    bool inited_ = false;

    WGPUBindGroupLayout tex_bind_layout_     = nullptr;
    WGPUSampler         sampler_             = nullptr;
    WGPUTexture         fallback_textures_[4] = {};
    WGPUTextureView     fallback_views_[4]    = {};
    WGPUBindGroup       tex_bind_group_      = nullptr;
    WGPUTextureView     cached_views_[4]     = {};

    vivid::gpu::VividSceneFragment  output_{};
    vivid::gpu::VividSceneFragment* child_ = nullptr;

    bool lazy_init(const VividGpuContext* gpu) {
        tex_bind_layout_ = vivid::gpu::create_pbr_texture_bind_layout(gpu->device);
        if (!tex_bind_layout_) return false;

        sampler_ = vivid::gpu::create_repeat_sampler(gpu->device, "Material3D Sampler");
        if (!sampler_) return false;

        // Fallback 1x1 RGBA8Unorm textures (same spec as Render3D)
        struct FallbackSpec { const char* label; uint8_t rgba[4]; };
        FallbackSpec specs[4] = {
            {"Material3D Fallback Albedo",   {255, 255, 255, 255}},
            {"Material3D Fallback Normal",   {128, 128, 255, 255}},
            {"Material3D Fallback R/M",      {255,   0,   0,   0}},
            {"Material3D Fallback Emission", {  0,   0,   0,   0}},
        };
        for (int i = 0; i < 4; ++i) {
            WGPUTextureDescriptor td{};
            td.label = vivid_sv(specs[i].label);
            td.size = {1, 1, 1};
            td.mipLevelCount = 1;
            td.sampleCount = 1;
            td.dimension = WGPUTextureDimension_2D;
            td.format = WGPUTextureFormat_RGBA8Unorm;
            td.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
            fallback_textures_[i] = wgpuDeviceCreateTexture(gpu->device, &td);

            WGPUTexelCopyTextureInfo dst{};
            dst.texture = fallback_textures_[i];
            dst.aspect = WGPUTextureAspect_All;
            WGPUTexelCopyBufferLayout layout_info{};
            layout_info.bytesPerRow = 4;
            layout_info.rowsPerImage = 1;
            WGPUExtent3D extent = {1, 1, 1};
            wgpuQueueWriteTexture(gpu->queue, &dst, specs[i].rgba, 4, &layout_info, &extent);

            WGPUTextureViewDescriptor vd{};
            vd.label = vivid_sv(specs[i].label);
            vd.format = WGPUTextureFormat_RGBA8Unorm;
            vd.dimension = WGPUTextureViewDimension_2D;
            vd.mipLevelCount = 1;
            vd.arrayLayerCount = 1;
            fallback_views_[i] = wgpuTextureCreateView(fallback_textures_[i], &vd);
        }

        inited_ = true;
        return true;
    }
};

VIVID_REGISTER(Material3D)
VIVID_THUMBNAIL(Material3D)

VIVID_DESCRIBE_REF_TYPE(vivid::gpu::VividSceneFragment)
