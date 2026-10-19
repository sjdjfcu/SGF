#include <kobuki_core/kobuki.hpp>
#include "velocity_lease.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

#include <ecl/geometry.hpp>
#include <ecl/linear_algebra.hpp>
#include <ecl/sigslots.hpp>

namespace {

std::atomic<bool> g_shutdown_requested(false);

void signal_handler(int) {
  g_shutdown_requested.store(true);
}

std::string json_escape(const std::string &value) {
  std::ostringstream out;
  for (char c : value) {
    switch (c) {
      case '\\':
        out << "\\\\";
        break;
      case '"':
        out << "\\\"";
        break;
      case '\n':
        out << "\\n";
        break;
      case '\r':
        out << "\\r";
        break;
      case '\t':
        out << "\\t";
        break;
      default:
        out << c;
        break;
    }
  }
  return out.str();
}

bool line_contains_cmd(const std::string &line, const std::string &cmd) {
  return line.find("\"cmd\":\"" + cmd + "\"") != std::string::npos ||
         line.find("\"cmd\": \"" + cmd + "\"") != std::string::npos;
}

double extract_double(const std::string &line, const std::string &key, double fallback) {
  const std::string quoted_key = "\"" + key + "\"";
  const std::size_t key_pos = line.find(quoted_key);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  const std::size_t colon_pos = line.find(':', key_pos + quoted_key.size());
  if (colon_pos == std::string::npos) {
    return fallback;
  }
  const char *start = line.c_str() + colon_pos + 1;
  char *end = nullptr;
  const double parsed = std::strtod(start, &end);
  if (end == start) {
    return fallback;
  }
  return parsed;
}

class Bridge {
 public:
  explicit Bridge(const std::string &device)
      : slot_stream_data_(&Bridge::process_stream_data, *this),
        last_status_time_(std::chrono::steady_clock::now()) {
    kobuki::Parameters parameters;
    parameters.sigslots_namespace = "/kobuki";
    parameters.device_port = device;
    parameters.enable_acceleration_limiter = true;
    kobuki_.init(parameters);
    kobuki_.enable();
    slot_stream_data_.connect("/kobuki/stream_data");
  }

  ~Bridge() {
    try {
      set_velocity(0.0, 0.0);
      kobuki_.setBaseControl(0.0, 0.0);
      kobuki_.disable();
    } catch (...) {
    }
  }

  void set_velocity(double linear, double angular, double timeout_s = 0.0) {
    std::lock_guard<std::mutex> lock(command_mutex_);
    command_linear_ = linear;
    command_angular_ = angular;
    command_lease_.renew(timeout_s);
  }

  void stop() {
    set_velocity(0.0, 0.0);
  }

  bool is_shutdown() const {
    return kobuki_.isShutdown();
  }

  void emit_ready() {
    std::cout << "{\"event\":\"ready\",\"alive\":" << (kobuki_.isAlive() ? "true" : "false")
              << ",\"enabled\":" << (kobuki_.isEnabled() ? "true" : "false")
              << ",\"velocity_timeout_supported\":true}" << std::endl;
  }

  void emit_error(const std::string &message) {
    std::cout << "{\"event\":\"error\",\"message\":\"" << json_escape(message) << "\"}" << std::endl;
  }

 private:
  void process_stream_data() {
    ecl::linear_algebra::Vector3d pose_update;
    ecl::linear_algebra::Vector3d pose_update_rates;
    kobuki_.updateOdometry(pose_update, pose_update_rates);
    ecl::concatenate_poses(pose_, pose_update);

    kobuki::CoreSensors::Data core = kobuki_.getCoreSensorData();
    bool safety_stop = core.bumper != 0 || core.cliff != 0 || core.wheel_drop != 0;
    double linear = 0.0;
    double angular = 0.0;
    {
      std::lock_guard<std::mutex> lock(command_mutex_);
      if (safety_stop || command_lease_.expired()) {
        command_linear_ = 0.0;
        command_angular_ = 0.0;
      }
      linear = command_linear_;
      angular = command_angular_;
    }
    kobuki_.setBaseControl(linear, angular);

    const auto now = std::chrono::steady_clock::now();
    const auto elapsed_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_status_time_).count();
    if (elapsed_ms >= 100 || safety_stop) {
      last_status_time_ = now;
      emit_status(core, safety_stop);
    }
  }

  void emit_status(const kobuki::CoreSensors::Data &core, bool safety_stop) {
    std::cout << "{\"event\":\"status\""
              << ",\"alive\":" << (kobuki_.isAlive() ? "true" : "false")
              << ",\"enabled\":" << (kobuki_.isEnabled() ? "true" : "false")
              << ",\"bumper\":" << static_cast<int>(core.bumper)
              << ",\"cliff\":" << static_cast<int>(core.cliff)
              << ",\"wheel_drop\":" << static_cast<int>(core.wheel_drop)
              << ",\"battery\":" << static_cast<int>(core.battery)
              << ",\"safety_stop\":" << (safety_stop ? "true" : "false")
              << ",\"x\":" << pose_[0]
              << ",\"y\":" << pose_[1]
              << ",\"heading\":" << pose_[2]
              << "}" << std::endl;
  }

  kobuki::Kobuki kobuki_;
  ecl::Slot<> slot_stream_data_;
  ecl::linear_algebra::Vector3d pose_;
  std::mutex command_mutex_;
  double command_linear_ = 0.0;
  double command_angular_ = 0.0;
  VelocityLease command_lease_;
  std::chrono::steady_clock::time_point last_status_time_;
};

}  // namespace

int main(int argc, char **argv) {
  std::string device = "/dev/kobuki";
  if (argc > 1) {
    device = argv[1];
  }
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  try {
    Bridge bridge(device);
    bridge.emit_ready();
    std::string line;
    while (!g_shutdown_requested.load() && !bridge.is_shutdown() && std::getline(std::cin, line)) {
      if (line_contains_cmd(line, "velocity")) {
        const double linear = extract_double(line, "linear", 0.0);
        const double angular = extract_double(line, "angular", 0.0);
        const double timeout_s = extract_double(line, "timeout_s", 0.0);
        bridge.set_velocity(linear, angular, timeout_s);
      } else if (line_contains_cmd(line, "stop")) {
        bridge.stop();
      } else if (line_contains_cmd(line, "quit")) {
        bridge.stop();
        break;
      }
    }
    bridge.stop();
  } catch (const std::exception &exc) {
    std::cout << "{\"event\":\"error\",\"message\":\"" << json_escape(exc.what()) << "\"}" << std::endl;
    std::cerr << exc.what() << std::endl;
    return 1;
  }
  return 0;
}
