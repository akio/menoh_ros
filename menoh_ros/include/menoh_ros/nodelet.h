#ifndef MENOH_ROS_NODELET_H_
#define MENOH_ROS_NODELET_H_


#include <string>
#include <vector>
#include <mutex>

#include "nodelet/nodelet.h"
#include "menoh/menoh.hpp"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/String.h"
#include "std_msgs/Float32MultiArray.h"

namespace menoh_ros {


class MenohNodelet : public nodelet::Nodelet {
 public:
  MenohNodelet();

  ~MenohNodelet() override = default;

  void onInit() override;

 private:
  void inputCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);

  std::unique_ptr<menoh::model> model_;

  std::string backend_name_;

  std::string input_variable_name_;
  std::string output_variable_name_;

  ros::Subscriber input_sub_;
  ros::Publisher output_pub_;
};

}  // namespace menoh_ros

#endif /*MENOH_ROS_NODELET_H_*/
