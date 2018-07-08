#ifndef MENOH_ROS_IO_NODELETS_H_
#define MENOH_ROS_IO_NODELETS_H_

#include <mutex>

#include "nodelet/nodelet.h"
#include "ros/ros.h"
#include "std_msgs/Float32MultiArray.h"
#include "sensor_msgs/Image.h"

namespace menoh_ros {


class ImageInputNodelet : public nodelet::Nodelet {
 public:
  ImageInputNodelet() = default;

  ~ImageInputNodelet() override = default;

  void onInit() override;

  void imageCallback(const sensor_msgs::Image::ConstPtr& msg);

 private:
  ros::Publisher pub_;
  ros::Subscriber sub_;
  double scale_{};
  int32_t input_size_{};
};

class CategoryOutputNodelet : public nodelet::Nodelet {
 public:
  CategoryOutputNodelet() = default;

  ~CategoryOutputNodelet() override = default;

  void onInit() override;

  void resultCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);

 private:
  ros::Subscriber sub_;
  ros::Publisher pub_;
  std::string category_names_path_;
};


}  // namespace menoh_ros

#endif /*(MENOH_ROS_IO_NODELETS_H_*/
