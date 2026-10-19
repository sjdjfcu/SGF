#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>

// Used under the bridge's command mutex
// A zero timeout never expires
class VelocityLease {
 public:
  using Clock = std::chrono::steady_clock;

  void renew(double timeout_s, Clock::time_point now = Clock::now()) {
    timeout_s_ = std::isfinite(timeout_s) ? std::max(0.0, timeout_s) : 0.0;
    updated_at_ = now;
  }

  bool expired(Clock::time_point now = Clock::now()) const {
    return timeout_s_ > 0.0 &&
        std::chrono::duration<double>(now - updated_at_).count() >= timeout_s_;
  }

 private:
  double timeout_s_ = 0.0;
  Clock::time_point updated_at_{};
};
