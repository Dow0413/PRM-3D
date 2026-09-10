#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <geometry_msgs/msg/transform_stamped.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <tf2/exceptions.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

namespace
{

struct BoxFilter
{
  double min_x{0.0};
  double max_x{0.0};
  double min_y{0.0};
  double max_y{0.0};
  double min_z{0.0};
  double max_z{0.0};
};

struct Point3D
{
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

struct GridIndex
{
  int64_t x{0};
  int64_t y{0};
  int64_t z{0};

  bool operator==(const GridIndex & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct GridIndexHash
{
  size_t operator()(const GridIndex & index) const
  {
    size_t seed = 0;
    combine(seed, index.x);
    combine(seed, index.y);
    combine(seed, index.z);
    return seed;
  }

  static void combine(size_t & seed, int64_t value)
  {
    const size_t hash_value = std::hash<int64_t>{}(value);
    seed ^= hash_value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
  }
};

template<typename T>
T readFieldValue(const uint8_t * data)
{
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

bool isFinitePoint(const pcl::PointXYZI & point)
{
  return std::isfinite(point.x) && std::isfinite(point.y) && std::isfinite(point.z);
}

bool pointInBox(const pcl::PointXYZI & point, const BoxFilter & box)
{
  return point.x >= box.min_x && point.x <= box.max_x &&
         point.y >= box.min_y && point.y <= box.max_y &&
         point.z >= box.min_z && point.z <= box.max_z;
}

bool pointWithinRadius(
  const pcl::PointXYZI & point,
  const Point3D & center,
  double radius_squared)
{
  const double dx = static_cast<double>(point.x) - center.x;
  const double dy = static_cast<double>(point.y) - center.y;
  const double dz = static_cast<double>(point.z) - center.z;
  return dx * dx + dy * dy + dz * dz <= radius_squared;
}

double squaredDistance(const pcl::PointXYZI & lhs, const pcl::PointXYZI & rhs)
{
  const double dx = static_cast<double>(lhs.x) - static_cast<double>(rhs.x);
  const double dy = static_cast<double>(lhs.y) - static_cast<double>(rhs.y);
  const double dz = static_cast<double>(lhs.z) - static_cast<double>(rhs.z);
  return dx * dx + dy * dy + dz * dz;
}

GridIndex pointGridIndex(const pcl::PointXYZI & point, double cell_size)
{
  return GridIndex{
    static_cast<int64_t>(std::floor(static_cast<double>(point.x) / cell_size)),
    static_cast<int64_t>(std::floor(static_cast<double>(point.y) / cell_size)),
    static_cast<int64_t>(std::floor(static_cast<double>(point.z) / cell_size))};
}

int findFieldIndex(
  const sensor_msgs::msg::PointCloud2 & cloud,
  const std::string & field_name)
{
  for (size_t i = 0; i < cloud.fields.size(); ++i) {
    if (cloud.fields[i].name == field_name) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

std::string fieldTypeName(uint8_t datatype)
{
  switch (datatype) {
    case sensor_msgs::msg::PointField::INT8:
      return "INT8";
    case sensor_msgs::msg::PointField::UINT8:
      return "UINT8";
    case sensor_msgs::msg::PointField::INT16:
      return "INT16";
    case sensor_msgs::msg::PointField::UINT16:
      return "UINT16";
    case sensor_msgs::msg::PointField::INT32:
      return "INT32";
    case sensor_msgs::msg::PointField::UINT32:
      return "UINT32";
    case sensor_msgs::msg::PointField::FLOAT32:
      return "FLOAT32";
    case sensor_msgs::msg::PointField::FLOAT64:
      return "FLOAT64";
    default:
      return "UNKNOWN";
  }
}

size_t fieldDatatypeSize(uint8_t datatype)
{
  switch (datatype) {
    case sensor_msgs::msg::PointField::INT8:
    case sensor_msgs::msg::PointField::UINT8:
      return 1;
    case sensor_msgs::msg::PointField::INT16:
    case sensor_msgs::msg::PointField::UINT16:
      return 2;
    case sensor_msgs::msg::PointField::INT32:
    case sensor_msgs::msg::PointField::UINT32:
    case sensor_msgs::msg::PointField::FLOAT32:
      return 4;
    case sensor_msgs::msg::PointField::FLOAT64:
      return 8;
    default:
      return 0;
  }
}

}  // namespace

class LocalCloudFilterNode : public rclcpp::Node
{
public:
  LocalCloudFilterNode()
  : Node("local_cloud_filter_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    declareParameters();
    loadParameters();
    sanitizeParameters();

    auto sensor_qos = rclcpp::SensorDataQoS().keep_last(
      static_cast<size_t>(queue_size_));
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_cloud_topic_, sensor_qos,
      std::bind(&LocalCloudFilterNode::cloudCallback, this, std::placeholders::_1));

    cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
      output_cloud_topic_, rclcpp::QoS(static_cast<size_t>(queue_size_)));

    RCLCPP_INFO(this->get_logger(), "local_cloud_filter_node started");
    RCLCPP_INFO(
      this->get_logger(), "Input: %s, output: %s, base_frame: %s, target_frame: %s",
      input_cloud_topic_.c_str(), output_cloud_topic_.c_str(),
      base_frame_.c_str(), target_frame_.c_str());
    if (enable_sensor_artifact_filter_) {
      RCLCPP_INFO(
        this->get_logger(),
        "Sensor artifact filter enabled: x[%.2f, %.2f], y[%.2f, %.2f], z[%.2f, %.2f]",
        sensor_artifact_box_.min_x, sensor_artifact_box_.max_x,
        sensor_artifact_box_.min_y, sensor_artifact_box_.max_y,
        sensor_artifact_box_.min_z, sensor_artifact_box_.max_z);
    }
  }

private:
  void declareParameters()
  {
    this->declare_parameter<std::string>("input_cloud_topic", "/front/rslidar_points");
    this->declare_parameter<std::string>("output_cloud_topic", "/local_cloud_map");
    this->declare_parameter<std::string>("target_frame", "map");
    this->declare_parameter<std::string>("base_frame", "base_link");

    this->declare_parameter<int>("queue_size", 10);
    this->declare_parameter<double>("tf_timeout_sec", 0.1);

    this->declare_parameter<double>("crop_min_x", -3.0);
    this->declare_parameter<double>("crop_max_x", 6.0);
    this->declare_parameter<double>("crop_min_y", -4.0);
    this->declare_parameter<double>("crop_max_y", 4.0);
    this->declare_parameter<double>("crop_min_z", -0.2);
    this->declare_parameter<double>("crop_max_z", 2.0);

    this->declare_parameter<double>("height_min_z", 0.05);
    this->declare_parameter<double>("height_max_z", 1.5);

    this->declare_parameter<double>("self_filter_min_x", -0.6);
    this->declare_parameter<double>("self_filter_max_x", 0.6);
    this->declare_parameter<double>("self_filter_min_y", -0.4);
    this->declare_parameter<double>("self_filter_max_y", 0.4);
    this->declare_parameter<double>("self_filter_min_z", -0.3);
    this->declare_parameter<double>("self_filter_max_z", 0.8);

    this->declare_parameter<double>("sensor_artifact_min_x", -0.45);
    this->declare_parameter<double>("sensor_artifact_max_x", 0.45);
    this->declare_parameter<double>("sensor_artifact_min_y", -0.45);
    this->declare_parameter<double>("sensor_artifact_max_y", 0.45);
    this->declare_parameter<double>("sensor_artifact_min_z", 0.35);
    this->declare_parameter<double>("sensor_artifact_max_z", 0.60);

    this->declare_parameter<double>("near_noise_radius", 0.30);
    this->declare_parameter<double>("near_noise_search_radius", 0.10);
    this->declare_parameter<int>("near_noise_min_neighbors", 3);

    this->declare_parameter<double>("voxel_leaf_size", 0.10);

    this->declare_parameter<bool>("enable_crop_filter", true);
    this->declare_parameter<bool>("enable_height_filter", true);
    this->declare_parameter<bool>("enable_self_filter", true);
    this->declare_parameter<bool>("enable_sensor_artifact_filter", false);
    this->declare_parameter<bool>("enable_near_noise_filter", false);
    this->declare_parameter<bool>("enable_voxel_filter", true);
  }

  void loadParameters()
  {
    input_cloud_topic_ = this->get_parameter("input_cloud_topic").as_string();
    output_cloud_topic_ = this->get_parameter("output_cloud_topic").as_string();
    target_frame_ = this->get_parameter("target_frame").as_string();
    base_frame_ = this->get_parameter("base_frame").as_string();

    queue_size_ = this->get_parameter("queue_size").as_int();
    tf_timeout_sec_ = this->get_parameter("tf_timeout_sec").as_double();

    crop_box_.min_x = this->get_parameter("crop_min_x").as_double();
    crop_box_.max_x = this->get_parameter("crop_max_x").as_double();
    crop_box_.min_y = this->get_parameter("crop_min_y").as_double();
    crop_box_.max_y = this->get_parameter("crop_max_y").as_double();
    crop_box_.min_z = this->get_parameter("crop_min_z").as_double();
    crop_box_.max_z = this->get_parameter("crop_max_z").as_double();

    height_min_z_ = this->get_parameter("height_min_z").as_double();
    height_max_z_ = this->get_parameter("height_max_z").as_double();

    self_box_.min_x = this->get_parameter("self_filter_min_x").as_double();
    self_box_.max_x = this->get_parameter("self_filter_max_x").as_double();
    self_box_.min_y = this->get_parameter("self_filter_min_y").as_double();
    self_box_.max_y = this->get_parameter("self_filter_max_y").as_double();
    self_box_.min_z = this->get_parameter("self_filter_min_z").as_double();
    self_box_.max_z = this->get_parameter("self_filter_max_z").as_double();

    sensor_artifact_box_.min_x =
      this->get_parameter("sensor_artifact_min_x").as_double();
    sensor_artifact_box_.max_x =
      this->get_parameter("sensor_artifact_max_x").as_double();
    sensor_artifact_box_.min_y =
      this->get_parameter("sensor_artifact_min_y").as_double();
    sensor_artifact_box_.max_y =
      this->get_parameter("sensor_artifact_max_y").as_double();
    sensor_artifact_box_.min_z =
      this->get_parameter("sensor_artifact_min_z").as_double();
    sensor_artifact_box_.max_z =
      this->get_parameter("sensor_artifact_max_z").as_double();

    near_noise_radius_ = this->get_parameter("near_noise_radius").as_double();
    near_noise_search_radius_ = this->get_parameter("near_noise_search_radius").as_double();
    near_noise_min_neighbors_ = this->get_parameter("near_noise_min_neighbors").as_int();

    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();

    enable_crop_filter_ = this->get_parameter("enable_crop_filter").as_bool();
    enable_height_filter_ = this->get_parameter("enable_height_filter").as_bool();
    enable_self_filter_ = this->get_parameter("enable_self_filter").as_bool();
    enable_sensor_artifact_filter_ =
      this->get_parameter("enable_sensor_artifact_filter").as_bool();
    enable_near_noise_filter_ = this->get_parameter("enable_near_noise_filter").as_bool();
    enable_voxel_filter_ = this->get_parameter("enable_voxel_filter").as_bool();
  }

  void sanitizeParameters()
  {
    if (queue_size_ <= 0) {
      RCLCPP_WARN(this->get_logger(), "queue_size must be positive. Using 10.");
      queue_size_ = 10;
    }

    if (tf_timeout_sec_ < 0.0) {
      RCLCPP_WARN(this->get_logger(), "tf_timeout_sec must be non-negative. Using 0.1.");
      tf_timeout_sec_ = 0.1;
    }

    normalizeBox(crop_box_, "crop");
    normalizeBox(self_box_, "self_filter");
    normalizeBox(sensor_artifact_box_, "sensor_artifact_filter");

    if (height_min_z_ > height_max_z_) {
      RCLCPP_WARN(this->get_logger(), "height_min_z is greater than height_max_z. Swapping.");
      std::swap(height_min_z_, height_max_z_);
    }

    if (near_noise_radius_ <= 0.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "near_noise_radius must be positive. Disabling near-noise filter.");
      enable_near_noise_filter_ = false;
    }
    near_noise_radius_squared_ = near_noise_radius_ * near_noise_radius_;

    if (near_noise_search_radius_ <= 0.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "near_noise_search_radius must be positive. Disabling near-noise filter.");
      enable_near_noise_filter_ = false;
    }
    near_noise_search_radius_squared_ = near_noise_search_radius_ * near_noise_search_radius_;

    if (near_noise_min_neighbors_ <= 0) {
      RCLCPP_WARN(
        this->get_logger(),
        "near_noise_min_neighbors must be positive. Disabling near-noise filter.");
      enable_near_noise_filter_ = false;
    }

    if (voxel_leaf_size_ <= 0.0) {
      RCLCPP_WARN(
        this->get_logger(),
        "voxel_leaf_size must be positive. Disabling voxel filter.");
      enable_voxel_filter_ = false;
    }
  }

  void normalizeBox(BoxFilter & box, const std::string & name)
  {
    if (box.min_x > box.max_x) {
      RCLCPP_WARN(this->get_logger(), "%s min_x is greater than max_x. Swapping.", name.c_str());
      std::swap(box.min_x, box.max_x);
    }
    if (box.min_y > box.max_y) {
      RCLCPP_WARN(this->get_logger(), "%s min_y is greater than max_y. Swapping.", name.c_str());
      std::swap(box.min_y, box.max_y);
    }
    if (box.min_z > box.max_z) {
      RCLCPP_WARN(this->get_logger(), "%s min_z is greater than max_z. Swapping.", name.c_str());
      std::swap(box.min_z, box.max_z);
    }
  }

  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!msg) {
      return;
    }

    if (msg->header.frame_id.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud has an empty frame_id");
      return;
    }

    if (msg->data.empty() || msg->width == 0 || msg->height == 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Received empty input cloud");
      publishEmptyCloud(msg->header.stamp);
      return;
    }

    sensor_msgs::msg::PointCloud2 cloud_base_msg;
    sensor_msgs::msg::PointCloud2 cloud_sensor_filtered_msg;
    const sensor_msgs::msg::PointCloud2 * cloud_for_transform = msg.get();
    if (enable_sensor_artifact_filter_) {
      if (!filterSensorArtifactCloud(*msg, cloud_sensor_filtered_msg)) {
        return;
      }
      cloud_for_transform = &cloud_sensor_filtered_msg;
    }

    if (!transformCloud(*cloud_for_transform, base_frame_, cloud_base_msg)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Failed to transform cloud from %s to %s",
        cloud_for_transform->header.frame_id.c_str(), base_frame_.c_str());
      return;
    }

    Point3D sensor_origin_base;
    if (
      enable_near_noise_filter_ &&
      !lookupSensorOriginInBaseFrame(*cloud_for_transform, sensor_origin_base))
    {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Failed to look up sensor origin from %s to %s for near-noise filtering",
        cloud_for_transform->header.frame_id.c_str(), base_frame_.c_str());
      return;
    }

    auto cloud_base = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    if (!convertToPclXYZI(cloud_base_msg, cloud_base)) {
      return;
    }

    auto cloud_filtered = filterCloudInBaseFrame(cloud_base, sensor_origin_base);
    auto cloud_downsampled = downsampleCloud(cloud_filtered);

    sensor_msgs::msg::PointCloud2 cloud_processed_base_msg;
    pcl::toROSMsg(*cloud_downsampled, cloud_processed_base_msg);
    cloud_processed_base_msg.header.stamp = msg->header.stamp;
    cloud_processed_base_msg.header.frame_id = base_frame_;

    sensor_msgs::msg::PointCloud2 cloud_out;
    if (!transformCloud(cloud_processed_base_msg, target_frame_, cloud_out)) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Failed to transform cloud from %s to %s",
        base_frame_.c_str(), target_frame_.c_str());
      return;
    }

    cloud_out.header.frame_id = target_frame_;
    cloud_out.header.stamp = msg->header.stamp;
    cloud_pub_->publish(cloud_out);
  }

  bool transformCloud(
    const sensor_msgs::msg::PointCloud2 & cloud_in,
    const std::string & target_frame,
    sensor_msgs::msg::PointCloud2 & cloud_out)
  {
    if (cloud_in.header.frame_id == target_frame) {
      cloud_out = cloud_in;
      cloud_out.header.frame_id = target_frame;
      return true;
    }

    try {
      const auto transform = tf_buffer_.lookupTransform(
        target_frame, cloud_in.header.frame_id, cloud_in.header.stamp,
        rclcpp::Duration::from_seconds(tf_timeout_sec_));
      tf2::doTransform(cloud_in, cloud_out, transform);
      cloud_out.header.stamp = cloud_in.header.stamp;
      cloud_out.header.frame_id = target_frame;
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_DEBUG(
        this->get_logger(), "TF transform failed from %s to %s: %s",
        cloud_in.header.frame_id.c_str(), target_frame.c_str(), ex.what());
      return false;
    } catch (const std::exception & ex) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "PointCloud2 transform error from %s to %s: %s",
        cloud_in.header.frame_id.c_str(), target_frame.c_str(), ex.what());
      return false;
    }
  }

  bool lookupSensorOriginInBaseFrame(
    const sensor_msgs::msg::PointCloud2 & cloud,
    Point3D & origin_base)
  {
    if (cloud.header.frame_id == base_frame_) {
      origin_base = Point3D{};
      return true;
    }

    try {
      const auto transform = tf_buffer_.lookupTransform(
        base_frame_, cloud.header.frame_id, cloud.header.stamp,
        rclcpp::Duration::from_seconds(tf_timeout_sec_));
      origin_base.x = transform.transform.translation.x;
      origin_base.y = transform.transform.translation.y;
      origin_base.z = transform.transform.translation.z;
      return true;
    } catch (const tf2::TransformException & ex) {
      RCLCPP_DEBUG(
        this->get_logger(), "Sensor-origin TF lookup failed from %s to %s: %s",
        cloud.header.frame_id.c_str(), base_frame_.c_str(), ex.what());
      return false;
    }
  }

  bool convertToPclXYZI(
    const sensor_msgs::msg::PointCloud2 & msg,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud)
  {
    const int x_index = findFieldIndex(msg, "x");
    const int y_index = findFieldIndex(msg, "y");
    const int z_index = findFieldIndex(msg, "z");

    if (x_index < 0 || y_index < 0 || z_index < 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud is missing one or more required fields: x, y, z");
      return false;
    }

    const auto & x_field = msg.fields[static_cast<size_t>(x_index)];
    const auto & y_field = msg.fields[static_cast<size_t>(y_index)];
    const auto & z_field = msg.fields[static_cast<size_t>(z_index)];
    if (
      x_field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
      y_field.datatype != sensor_msgs::msg::PointField::FLOAT32 ||
      z_field.datatype != sensor_msgs::msg::PointField::FLOAT32)
    {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud x/y/z fields must be FLOAT32. Got x=%s, y=%s, z=%s",
        fieldTypeName(x_field.datatype).c_str(),
        fieldTypeName(y_field.datatype).c_str(),
        fieldTypeName(z_field.datatype).c_str());
      return false;
    }

    const int intensity_index = findFieldIndex(msg, "intensity");
    const sensor_msgs::msg::PointField * intensity_field = nullptr;
    if (intensity_index >= 0) {
      intensity_field = &msg.fields[static_cast<size_t>(intensity_index)];
      if (!validateFieldBounds(msg, *intensity_field, "intensity")) {
        intensity_field = nullptr;
      }
    } else {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 30000,
        "Input cloud has no intensity field. Output intensity will be 0.0");
    }

    if (
      !validateFieldBounds(msg, x_field, "x") ||
      !validateFieldBounds(msg, y_field, "y") ||
      !validateFieldBounds(msg, z_field, "z") ||
      !validatePointCloudLayout(msg))
    {
      return false;
    }

    cloud->clear();
    cloud->reserve(static_cast<size_t>(msg.width) * static_cast<size_t>(msg.height));

    for (uint32_t row = 0; row < msg.height; ++row) {
      const uint8_t * row_data = msg.data.data() + static_cast<size_t>(row) * msg.row_step;
      for (uint32_t col = 0; col < msg.width; ++col) {
        const uint8_t * point_data = row_data + static_cast<size_t>(col) * msg.point_step;

        pcl::PointXYZI point;
        point.x = readFieldValue<float>(point_data + x_field.offset);
        point.y = readFieldValue<float>(point_data + y_field.offset);
        point.z = readFieldValue<float>(point_data + z_field.offset);
        point.intensity = intensity_field ? readIntensity(point_data, *intensity_field) : 0.0F;

        if (isFinitePoint(point)) {
          cloud->push_back(point);
        }
      }
    }

    cloud->width = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = true;
    cloud->header.frame_id = msg.header.frame_id;

    if (cloud->empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud has no finite xyz points after conversion");
    }

    return true;
  }

  bool filterSensorArtifactCloud(
    const sensor_msgs::msg::PointCloud2 & msg,
    sensor_msgs::msg::PointCloud2 & filtered_msg)
  {
    auto cloud_sensor = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    if (!convertToPclXYZI(msg, cloud_sensor)) {
      return false;
    }

    auto cloud_filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud_filtered->reserve(cloud_sensor->size());
    for (const auto & point : cloud_sensor->points) {
      if (!isFinitePoint(point)) {
        continue;
      }

      if (pointInBox(point, sensor_artifact_box_)) {
        continue;
      }

      cloud_filtered->push_back(point);
    }

    cloud_filtered->width = static_cast<uint32_t>(cloud_filtered->size());
    cloud_filtered->height = 1;
    cloud_filtered->is_dense = true;
    cloud_filtered->header.frame_id = msg.header.frame_id;

    pcl::toROSMsg(*cloud_filtered, filtered_msg);
    filtered_msg.header = msg.header;
    return true;
  }

  bool validateFieldBounds(
    const sensor_msgs::msg::PointCloud2 & msg,
    const sensor_msgs::msg::PointField & field,
    const std::string & field_name)
  {
    const size_t datatype_size = fieldDatatypeSize(field.datatype);
    const size_t count = std::max<uint32_t>(field.count, 1U);
    const size_t field_size = datatype_size * count;

    if (datatype_size == 0 || field_size == 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud field %s has unsupported datatype %s",
        field_name.c_str(), fieldTypeName(field.datatype).c_str());
      return false;
    }

    if (static_cast<size_t>(field.offset) + field_size > msg.point_step) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud field %s offset exceeds point_step", field_name.c_str());
      return false;
    }

    return true;
  }

  bool validatePointCloudLayout(const sensor_msgs::msg::PointCloud2 & msg)
  {
    if (msg.point_step == 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud point_step is 0");
      return false;
    }

    const size_t width = static_cast<size_t>(msg.width);
    const size_t height = static_cast<size_t>(msg.height);
    const size_t min_row_step = width * static_cast<size_t>(msg.point_step);
    if (msg.row_step < min_row_step) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud row_step is smaller than width * point_step");
      return false;
    }

    if (height == 0) {
      return false;
    }

    const size_t min_data_size =
      (height - 1) * static_cast<size_t>(msg.row_step) + min_row_step;
    if (msg.data.size() < min_data_size) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 5000,
        "Input cloud data size is smaller than expected from layout");
      return false;
    }

    return true;
  }

  float readIntensity(
    const uint8_t * point_data,
    const sensor_msgs::msg::PointField & field)
  {
    const uint8_t * data = point_data + field.offset;

    switch (field.datatype) {
      case sensor_msgs::msg::PointField::INT8:
        return static_cast<float>(readFieldValue<int8_t>(data));
      case sensor_msgs::msg::PointField::UINT8:
        return static_cast<float>(readFieldValue<uint8_t>(data));
      case sensor_msgs::msg::PointField::INT16:
        return static_cast<float>(readFieldValue<int16_t>(data));
      case sensor_msgs::msg::PointField::UINT16:
        return static_cast<float>(readFieldValue<uint16_t>(data));
      case sensor_msgs::msg::PointField::INT32:
        return static_cast<float>(readFieldValue<int32_t>(data));
      case sensor_msgs::msg::PointField::UINT32:
        return static_cast<float>(readFieldValue<uint32_t>(data));
      case sensor_msgs::msg::PointField::FLOAT32:
        return readFieldValue<float>(data);
      case sensor_msgs::msg::PointField::FLOAT64:
        return static_cast<float>(readFieldValue<double>(data));
      default:
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 30000,
          "Unsupported intensity field type %s. Using 0.0",
          fieldTypeName(field.datatype).c_str());
        return 0.0F;
    }
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr filterCloudInBaseFrame(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_in,
    const Point3D & sensor_origin_base)
  {
    auto cloud_out = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    cloud_out->reserve(cloud_in->size());

    const auto near_sparse_noise = markNearSparseNoisePoints(cloud_in, sensor_origin_base);

    for (size_t point_index = 0; point_index < cloud_in->points.size(); ++point_index) {
      const auto & point = cloud_in->points[point_index];
      if (!isFinitePoint(point)) {
        continue;
      }

      if (enable_near_noise_filter_ && near_sparse_noise[point_index]) {
        continue;
      }

      if (enable_crop_filter_ && !pointInBox(point, crop_box_)) {
        continue;
      }

      if (enable_self_filter_ && pointInBox(point, self_box_)) {
        continue;
      }

      if (enable_height_filter_ && (point.z < height_min_z_ || point.z > height_max_z_)) {
        continue;
      }

      cloud_out->push_back(point);
    }

    cloud_out->width = static_cast<uint32_t>(cloud_out->size());
    cloud_out->height = 1;
    cloud_out->is_dense = true;
    cloud_out->header.frame_id = base_frame_;
    return cloud_out;
  }

  std::vector<uint8_t> markNearSparseNoisePoints(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud,
    const Point3D & sensor_origin_base)
  {
    std::vector<uint8_t> sparse_noise(cloud->points.size(), 0U);
    if (!enable_near_noise_filter_ || cloud->empty()) {
      return sparse_noise;
    }

    std::unordered_map<GridIndex, std::vector<size_t>, GridIndexHash> point_grid;
    point_grid.reserve(cloud->points.size());
    std::vector<size_t> near_indices;

    for (size_t point_index = 0; point_index < cloud->points.size(); ++point_index) {
      const auto & point = cloud->points[point_index];
      if (!isFinitePoint(point)) {
        continue;
      }

      point_grid[pointGridIndex(point, near_noise_search_radius_)].push_back(point_index);
      if (pointWithinRadius(point, sensor_origin_base, near_noise_radius_squared_)) {
        near_indices.push_back(point_index);
      }
    }

    for (const size_t point_index : near_indices) {
      const auto & point = cloud->points[point_index];
      const auto center_cell = pointGridIndex(point, near_noise_search_radius_);
      int neighbor_count = 0;

      for (int64_t dx = -1; dx <= 1; ++dx) {
        for (int64_t dy = -1; dy <= 1; ++dy) {
          for (int64_t dz = -1; dz <= 1; ++dz) {
            const GridIndex neighbor_cell{
              center_cell.x + dx, center_cell.y + dy, center_cell.z + dz};
            const auto cell_points = point_grid.find(neighbor_cell);
            if (cell_points == point_grid.end()) {
              continue;
            }

            for (const size_t neighbor_index : cell_points->second) {
              if (neighbor_index == point_index) {
                continue;
              }

              if (
                squaredDistance(point, cloud->points[neighbor_index]) <=
                near_noise_search_radius_squared_)
              {
                ++neighbor_count;
                if (neighbor_count >= near_noise_min_neighbors_) {
                  break;
                }
              }
            }

            if (neighbor_count >= near_noise_min_neighbors_) {
              break;
            }
          }

          if (neighbor_count >= near_noise_min_neighbors_) {
            break;
          }
        }

        if (neighbor_count >= near_noise_min_neighbors_) {
          break;
        }
      }

      if (neighbor_count < near_noise_min_neighbors_) {
        sparse_noise[point_index] = 1U;
      }
    }

    return sparse_noise;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr downsampleCloud(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_in)
  {
    if (!enable_voxel_filter_ || cloud_in->empty()) {
      return cloud_in;
    }

    auto cloud_out = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
    pcl::VoxelGrid<pcl::PointXYZI> voxel_filter;
    voxel_filter.setInputCloud(cloud_in);
    const auto leaf_size = static_cast<float>(voxel_leaf_size_);
    voxel_filter.setLeafSize(leaf_size, leaf_size, leaf_size);
    voxel_filter.filter(*cloud_out);
    cloud_out->header.frame_id = base_frame_;
    return cloud_out;
  }

  void publishEmptyCloud(const rclcpp::Time & stamp)
  {
    pcl::PointCloud<pcl::PointXYZI> empty_cloud;
    empty_cloud.header.frame_id = target_frame_;
    empty_cloud.width = 0;
    empty_cloud.height = 1;
    empty_cloud.is_dense = true;

    sensor_msgs::msg::PointCloud2 cloud_out;
    pcl::toROSMsg(empty_cloud, cloud_out);
    cloud_out.header.stamp = stamp;
    cloud_out.header.frame_id = target_frame_;
    cloud_pub_->publish(cloud_out);
  }

  std::string input_cloud_topic_;
  std::string output_cloud_topic_;
  std::string target_frame_;
  std::string base_frame_;

  int queue_size_{10};
  double tf_timeout_sec_{0.1};

  BoxFilter crop_box_;
  BoxFilter self_box_;
  double height_min_z_{0.05};
  double height_max_z_{1.5};
  BoxFilter sensor_artifact_box_;
  double near_noise_radius_{0.30};
  double near_noise_radius_squared_{0.09};
  double near_noise_search_radius_{0.10};
  double near_noise_search_radius_squared_{0.01};
  int near_noise_min_neighbors_{3};
  double voxel_leaf_size_{0.10};

  bool enable_crop_filter_{true};
  bool enable_height_filter_{true};
  bool enable_self_filter_{true};
  bool enable_sensor_artifact_filter_{false};
  bool enable_near_noise_filter_{false};
  bool enable_voxel_filter_{true};

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LocalCloudFilterNode>());
  rclcpp::shutdown();
  return 0;
}
