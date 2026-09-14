#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace smpc_lane_change {

struct Point2D {
  double x{0.0};
  double y{0.0};
};

struct Projection {
  bool valid{false};
  double s{0.0};
  double d{0.0};
  double yaw{0.0};
  double distance{std::numeric_limits<double>::infinity()};
  std::size_t segment_index{0};
};

struct LaneSample {
  bool valid{false};
  double x{0.0};
  double y{0.0};
  double s{0.0};
  double d{0.0};
  double yaw{0.0};
};

class LanePath {
 public:
  bool loadCsv(const std::string& file_path) {
    points_.clear();
    cumulative_s_.clear();
    file_path_ = file_path;

    std::ifstream file(file_path);
    if (!file.is_open()) return false;

    std::string line;
    while (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string cell;
      std::vector<std::string> cells;
      while (std::getline(ss, cell, ',')) cells.push_back(cell);
      if (cells.size() < 3) continue;

      try {
        // MPC waypoint CSV: index,x,y,mission_state,heading,curvature
        points_.push_back({std::stod(cells[1]), std::stod(cells[2])});
      } catch (...) {
        continue;
      }
    }

    if (points_.size() < 2) return false;
    cumulative_s_.resize(points_.size(), 0.0);
    for (std::size_t i = 1; i < points_.size(); ++i) {
      cumulative_s_[i] = cumulative_s_[i - 1] +
          std::hypot(points_[i].x - points_[i - 1].x,
                     points_[i].y - points_[i - 1].y);
    }
    return true;
  }

  Projection project(double qx, double qy) const {
    Projection best;
    if (points_.size() < 2) return best;

    double best_d2 = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i + 1 < points_.size(); ++i) {
      const double vx = points_[i + 1].x - points_[i].x;
      const double vy = points_[i + 1].y - points_[i].y;
      const double len2 = vx * vx + vy * vy;
      if (len2 < 1e-10) continue;

      const double rx = qx - points_[i].x;
      const double ry = qy - points_[i].y;
      const double t = std::clamp((rx * vx + ry * vy) / len2, 0.0, 1.0);
      const double px = points_[i].x + t * vx;
      const double py = points_[i].y + t * vy;
      const double ex = qx - px;
      const double ey = qy - py;
      const double d2 = ex * ex + ey * ey;

      if (d2 < best_d2) {
        best_d2 = d2;
        const double seg_len = std::sqrt(len2);
        const double cross = vx * ey - vy * ex;
        best.valid = true;
        best.s = cumulative_s_[i] + t * seg_len;
        best.d = (cross >= 0.0 ? 1.0 : -1.0) * std::sqrt(d2);
        best.yaw = std::atan2(vy, vx);
        best.distance = std::sqrt(d2);
        best.segment_index = i;
      }
    }
    return best;
  }

  // Bounded endpoint-tangent projection for perception/prediction.  The MPC
  // path itself remains unchanged.
  Projection projectExtended(double qx, double qy,
                             double before_start_m,
                             double after_end_m = 0.0) const {
    Projection best;
    if (points_.size() < 2) return best;
    double best_d2 = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i + 1 < points_.size(); ++i) {
      const double vx = points_[i + 1].x - points_[i].x;
      const double vy = points_[i + 1].y - points_[i].y;
      const double len2 = vx * vx + vy * vy;
      if (len2 < 1e-10) continue;
      const double seg_len = std::sqrt(len2);
      const double raw_t = ((qx - points_[i].x) * vx +
                            (qy - points_[i].y) * vy) / len2;
      const double lower = i == 0
          ? -std::max(0.0, before_start_m) / seg_len : 0.0;
      const double upper = i + 2 == points_.size()
          ? 1.0 + std::max(0.0, after_end_m) / seg_len : 1.0;
      const double t = std::clamp(raw_t, lower, upper);
      const double px = points_[i].x + t * vx;
      const double py = points_[i].y + t * vy;
      const double ex = qx - px;
      const double ey = qy - py;
      const double d2 = ex * ex + ey * ey;
      if (d2 >= best_d2) continue;
      best_d2 = d2;
      best.valid = true;
      best.s = cumulative_s_[i] + t * seg_len;
      best.d = (vx * ey - vy * ex >= 0.0 ? 1.0 : -1.0) * std::sqrt(d2);
      best.yaw = std::atan2(vy, vx);
      best.distance = std::sqrt(d2);
      best.segment_index = i;
    }
    return best;
  }

  LaneSample sample(double query_s, double lateral_d = 0.0) const {
    LaneSample out;
    if (points_.size() < 2 || cumulative_s_.size() != points_.size()) return out;

    const double clamped_s = std::clamp(query_s, 0.0, length());
    auto upper = std::lower_bound(cumulative_s_.begin(), cumulative_s_.end(), clamped_s);
    std::size_t i = 0;
    if (upper == cumulative_s_.begin()) {
      i = 0;
    } else if (upper == cumulative_s_.end()) {
      i = points_.size() - 2;
    } else {
      i = static_cast<std::size_t>(std::distance(cumulative_s_.begin(), upper) - 1);
      i = std::min(i, points_.size() - 2);
    }

    const double vx = points_[i + 1].x - points_[i].x;
    const double vy = points_[i + 1].y - points_[i].y;
    const double seg_len = std::hypot(vx, vy);
    if (seg_len < 1e-9) return out;

    const double t = std::clamp((clamped_s - cumulative_s_[i]) / seg_len, 0.0, 1.0);
    const double cx = points_[i].x + t * vx;
    const double cy = points_[i].y + t * vy;
    const double yaw = std::atan2(vy, vx);
    const double nx = -std::sin(yaw);
    const double ny = std::cos(yaw);

    out.valid = true;
    out.x = cx + lateral_d * nx;
    out.y = cy + lateral_d * ny;
    out.s = clamped_s;
    out.d = lateral_d;
    out.yaw = yaw;
    return out;
  }

  LaneSample sampleExtended(double query_s, double lateral_d,
                            double before_start_m,
                            double after_end_m = 0.0) const {
    if (query_s >= 0.0 && query_s <= length()) return sample(query_s, lateral_d);
    LaneSample out;
    if (points_.size() < 2 ||
        query_s < -std::max(0.0, before_start_m) ||
        query_s > length() + std::max(0.0, after_end_m)) {
      return out;
    }
    const std::size_t i = query_s < 0.0 ? 0 : points_.size() - 2;
    const double vx = points_[i + 1].x - points_[i].x;
    const double vy = points_[i + 1].y - points_[i].y;
    const double seg_len = std::hypot(vx, vy);
    if (seg_len < 1e-9) return out;
    const double t = (query_s - cumulative_s_[i]) / seg_len;
    const double yaw = std::atan2(vy, vx);
    out.valid = true;
    out.x = points_[i].x + t * vx - lateral_d * std::sin(yaw);
    out.y = points_[i].y + t * vy + lateral_d * std::cos(yaw);
    out.s = query_s;
    out.d = lateral_d;
    out.yaw = yaw;
    return out;
  }

  bool empty() const { return points_.empty(); }
  double length() const { return cumulative_s_.empty() ? 0.0 : cumulative_s_.back(); }
  const std::vector<Point2D>& points() const { return points_; }
  const std::string& filePath() const { return file_path_; }

 private:
  std::string file_path_;
  std::vector<Point2D> points_;
  std::vector<double> cumulative_s_;
};

inline double wrapToPi(double angle) {
  return std::atan2(std::sin(angle), std::cos(angle));
}

}  // namespace smpc_lane_change
