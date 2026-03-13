#include "operator_api/operator.h"
#include "operator_api/gpu_operator.h"
#include "operator_api/gpu_3d.h"
#include <cstring>
#include <cmath>

// =============================================================================
// Transform3D — scene-in, scene-out with TRS transform
// =============================================================================

struct Transform3D : vivid::GpuOperatorBase {
    static constexpr const char* kName   = "Transform3D";
    static constexpr bool kTimeDependent = false;

    vivid::Param<float> pos_x   {"pos_x",   0.0f, -50.0f, 50.0f};
    vivid::Param<float> pos_y   {"pos_y",   0.0f, -50.0f, 50.0f};
    vivid::Param<float> pos_z   {"pos_z",   0.0f, -50.0f, 50.0f};
    vivid::Param<float> rot_x   {"rot_x",   0.0f, -6.283f, 6.283f};
    vivid::Param<float> rot_y   {"rot_y",   0.0f, -6.283f, 6.283f};
    vivid::Param<float> rot_z   {"rot_z",   0.0f, -6.283f, 6.283f};
    vivid::Param<float> scale_x {"scale_x", 1.0f,  0.01f, 50.0f};
    vivid::Param<float> scale_y {"scale_y", 1.0f,  0.01f, 50.0f};
    vivid::Param<float> scale_z {"scale_z", 1.0f,  0.01f, 50.0f};

    Transform3D() {
        vivid::semantic_tag(pos_x, "position_xyz");
        vivid::semantic_shape(pos_x, "scalar");
        vivid::semantic_intent(pos_x, "position_x");
        vivid::semantic_tag(pos_y, "position_xyz");
        vivid::semantic_shape(pos_y, "scalar");
        vivid::semantic_intent(pos_y, "position_y");
        vivid::semantic_tag(pos_z, "position_xyz");
        vivid::semantic_shape(pos_z, "scalar");
        vivid::semantic_intent(pos_z, "position_z");

        vivid::semantic_tag(rot_x, "rotation_radians");
        vivid::semantic_shape(rot_x, "scalar");
        vivid::semantic_unit(rot_x, "rad");
        vivid::semantic_intent(rot_x, "rotation_x");
        vivid::semantic_tag(rot_y, "rotation_radians");
        vivid::semantic_shape(rot_y, "scalar");
        vivid::semantic_unit(rot_y, "rad");
        vivid::semantic_intent(rot_y, "rotation_y");
        vivid::semantic_tag(rot_z, "rotation_radians");
        vivid::semantic_shape(rot_z, "scalar");
        vivid::semantic_unit(rot_z, "rad");
        vivid::semantic_intent(rot_z, "rotation_z");

        vivid::semantic_tag(scale_x, "scale_xyz");
        vivid::semantic_shape(scale_x, "scalar");
        vivid::semantic_intent(scale_x, "scale_x");
        vivid::semantic_tag(scale_y, "scale_xyz");
        vivid::semantic_shape(scale_y, "scalar");
        vivid::semantic_intent(scale_y, "scale_y");
        vivid::semantic_tag(scale_z, "scale_xyz");
        vivid::semantic_shape(scale_z, "scalar");
        vivid::semantic_intent(scale_z, "scale_z");
    }

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        vivid::param_group(pos_x, "Transform");
        vivid::param_group(pos_y, "Transform");
        vivid::param_group(pos_z, "Transform");
        vivid::param_group(rot_x, "Transform");
        vivid::param_group(rot_y, "Transform");
        vivid::param_group(rot_z, "Transform");
        vivid::param_group(scale_x, "Transform");
        vivid::param_group(scale_y, "Transform");
        vivid::param_group(scale_z, "Transform");

        out.push_back(&pos_x);
        out.push_back(&pos_y);
        out.push_back(&pos_z);
        out.push_back(&rot_x);
        out.push_back(&rot_y);
        out.push_back(&rot_z);
        out.push_back(&scale_x);
        out.push_back(&scale_y);
        out.push_back(&scale_z);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_INPUT));
        out.push_back(vivid::gpu::scene_port("scene", VIVID_PORT_OUTPUT));
    }

    void process_gpu(const VividGpuContext* ctx) override {
        // No input scene → no output
        bool has_input = ctx->custom_input_count > 0 &&
                         vivid::gpu::scene_input(ctx, 0) != nullptr;
        if (!has_input) return;

        // Build TRS matrix: T * Rz * Ry * Rx * S (same order as Shape3D)
        mat4x4 S, tmp;
        mat4x4_identity(S);
        mat4x4_scale_aniso(S, S, scale_x.value, scale_y.value, scale_z.value);
        mat4x4_rotate_X(tmp, S, rot_x.value);
        mat4x4_rotate_Y(S, tmp, rot_y.value);
        mat4x4_rotate_Z(tmp, S, rot_z.value);

        mat4x4 T;
        mat4x4_translate(T, pos_x.value, pos_y.value, pos_z.value);
        mat4x4_mul(output_.model_matrix, T, tmp);

        // No geometry on this fragment — just a transform wrapper
        output_.vertex_buffer   = nullptr;
        output_.vertex_buf_size = 0;
        output_.index_buffer    = nullptr;
        output_.index_count     = 0;
        output_.pipeline        = nullptr;
        output_.material_binds  = nullptr;
        output_.fragment_type   = vivid::gpu::VividSceneFragment::GEOMETRY;

        // Wrap input as child
        child_ = vivid::gpu::scene_input(ctx, 0);
        output_.children    = &child_;
        output_.child_count = 1;

        ctx->custom_outputs[0] = &output_;
    }

private:
    vivid::gpu::VividSceneFragment  output_{};
    vivid::gpu::VividSceneFragment* child_ = nullptr;
};

VIVID_REGISTER(Transform3D)

VIVID_DESCRIBE_REF_TYPE(vivid::gpu::VividSceneFragment)
