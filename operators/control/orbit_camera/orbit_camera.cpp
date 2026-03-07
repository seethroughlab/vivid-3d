#include "operator_api/operator.h"
#include "operator_api/input_state.h"
#include <cmath>

struct OrbitCamera : vivid::OperatorBase {
    static constexpr const char* kName   = "OrbitCamera";
    static constexpr VividDomain kDomain = VIVID_DOMAIN_CONTROL;
    static constexpr bool kTimeDependent = true;

    // Orbit
    vivid::Param<float> distance_p{"distance", 5.0f, 0.1f, 200.0f};
    vivid::Param<float> azimuth_p{"azimuth", 0.0f, -180.0f, 180.0f};
    vivid::Param<float> elevation_p{"elevation", 20.0f, -89.0f, 89.0f};
    // Target
    vivid::Param<float> target_x_p{"target_x", 0.0f, -50.0f, 50.0f};
    vivid::Param<float> target_y_p{"target_y", 0.0f, -50.0f, 50.0f};
    vivid::Param<float> target_z_p{"target_z", 0.0f, -50.0f, 50.0f};
    // Sensitivity
    vivid::Param<float> orbit_sens{"orbit_sensitivity", 3.0f, 0.1f, 20.0f};
    vivid::Param<float> pan_sens{"pan_sensitivity", 2.0f, 0.1f, 20.0f};
    vivid::Param<float> zoom_sens{"zoom_sensitivity", 0.1f, 0.01f, 1.0f};
    // Limits
    vivid::Param<float> min_dist{"min_distance", 0.1f, 0.01f, 10.0f};
    vivid::Param<float> max_dist{"max_distance", 100.0f, 1.0f, 1000.0f};

    // Internal state
    float azimuth_rad_ = 0.0f;
    float elevation_rad_ = 0.0f;
    float distance_ = 5.0f;
    float tgt_[3] = {0.0f, 0.0f, 0.0f};
    float prev_mx_ = 0.0f;
    float prev_my_ = 0.0f;
    bool first_frame_ = true;
    bool first_input_ = true;

    void collect_params(std::vector<vivid::ParamBase*>& out) override {
        vivid::param_group(distance_p, "Orbit");
        vivid::param_group(azimuth_p, "Orbit");
        vivid::param_group(elevation_p, "Orbit");
        vivid::param_group(target_x_p, "Target");
        vivid::param_group(target_y_p, "Target");
        vivid::param_group(target_z_p, "Target");
        vivid::param_group(orbit_sens, "Sensitivity");
        vivid::param_group(pan_sens, "Sensitivity");
        vivid::param_group(zoom_sens, "Sensitivity");
        vivid::param_group(min_dist, "Limits");
        vivid::param_group(max_dist, "Limits");

        out.push_back(&distance_p);
        out.push_back(&azimuth_p);
        out.push_back(&elevation_p);
        out.push_back(&target_x_p);
        out.push_back(&target_y_p);
        out.push_back(&target_z_p);
        out.push_back(&orbit_sens);
        out.push_back(&pan_sens);
        out.push_back(&zoom_sens);
        out.push_back(&min_dist);
        out.push_back(&max_dist);
    }

    void collect_ports(std::vector<VividPortDescriptor>& out) override {
        out.push_back({"cam_x",    VIVID_PORT_CONTROL_FLOAT, VIVID_PORT_OUTPUT});
        out.push_back({"cam_y",    VIVID_PORT_CONTROL_FLOAT, VIVID_PORT_OUTPUT});
        out.push_back({"cam_z",    VIVID_PORT_CONTROL_FLOAT, VIVID_PORT_OUTPUT});
        out.push_back({"target_x", VIVID_PORT_CONTROL_FLOAT, VIVID_PORT_OUTPUT});
        out.push_back({"target_y", VIVID_PORT_CONTROL_FLOAT, VIVID_PORT_OUTPUT});
        out.push_back({"target_z", VIVID_PORT_CONTROL_FLOAT, VIVID_PORT_OUTPUT});
    }

    void write_outputs(const VividProcessContext* ctx) {
        float azim = azimuth_rad_;
        float elev = elevation_rad_;
        float ce = cosf(elev);
        float se = sinf(elev);
        float sa = sinf(azim);
        float ca = cosf(azim);

        ctx->output_values[0] = tgt_[0] + distance_ * ce * sa;
        ctx->output_values[1] = tgt_[1] + distance_ * se;
        ctx->output_values[2] = tgt_[2] + distance_ * ce * ca;
        ctx->output_values[3] = tgt_[0];
        ctx->output_values[4] = tgt_[1];
        ctx->output_values[5] = tgt_[2];
    }

    void process(const VividProcessContext* ctx) override {
        static constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
        static constexpr float kMaxElev  = 89.0f * kDegToRad;

        // Initialize orbital state from params on first call (no input needed)
        if (first_frame_) {
            first_frame_ = false;
            azimuth_rad_   = azimuth_p.value * kDegToRad;
            elevation_rad_ = elevation_p.value * kDegToRad;
            distance_      = distance_p.value;
            tgt_[0] = target_x_p.value;
            tgt_[1] = target_y_p.value;
            tgt_[2] = target_z_p.value;
        }

        const VividInputState* input = vivid_input(ctx);

        if (!input) {
            write_outputs(ctx);
            return;
        }

        // Capture initial mouse pos on first frame with input (no orbit delta)
        if (first_input_) {
            first_input_ = false;
            prev_mx_ = input->mouse_x;
            prev_my_ = input->mouse_y;
            write_outputs(ctx);
            return;
        }

        float dx = input->mouse_x - prev_mx_;
        float dy = input->mouse_y - prev_my_;

        // Left button: orbit
        if (input->buttons_held & (1 << 0)) {
            azimuth_rad_   -= dx * orbit_sens.value;
            elevation_rad_ += dy * orbit_sens.value;
            if (elevation_rad_ >  kMaxElev) elevation_rad_ =  kMaxElev;
            if (elevation_rad_ < -kMaxElev) elevation_rad_ = -kMaxElev;
        }

        // Right or middle button: pan
        if (input->buttons_held & ((1 << 1) | (1 << 2))) {
            float sa = sinf(azimuth_rad_);
            float ca = cosf(azimuth_rad_);
            float se = sinf(elevation_rad_);
            float ce = cosf(elevation_rad_);

            // Camera right vector (Y-up)
            float rx = ca, ry = 0.0f, rz = -sa;
            // Camera up vector
            float ux = -se * sa, uy = ce, uz = -se * ca;

            float scale = pan_sens.value * distance_;
            tgt_[0] += (-dx * rx + dy * ux) * scale;
            tgt_[1] += (-dx * ry + dy * uy) * scale;
            tgt_[2] += (-dx * rz + dy * uz) * scale;
        }

        // Scroll: zoom
        for (uint32_t i = 0; i < input->event_count; ++i) {
            if (input->events[i].type == VIVID_INPUT_MOUSE_SCROLL) {
                distance_ *= (1.0f - input->events[i].scroll_dy * zoom_sens.value);
                if (distance_ < min_dist.value) distance_ = min_dist.value;
                if (distance_ > max_dist.value) distance_ = max_dist.value;
            }
        }

        prev_mx_ = input->mouse_x;
        prev_my_ = input->mouse_y;

        write_outputs(ctx);
    }
};

VIVID_REGISTER(OrbitCamera)
