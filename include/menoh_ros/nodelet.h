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

class InputPluginBase {
 public:
  InputPluginBase() = default;
  virtual ~InputPluginBase() = default;

  InputPluginBase(const InputPluginBase&) = delete;
  InputPluginBase& operator=(const InputPluginBase&) = delete;

  virtual void initialize(ros::NodeHandle& nh,
                          std::vector<int32_t>& dst_dims) = 0;

  virtual bool execute(menoh::variable var) = 0;
};

//
class OutputPluginBase {
 public:
  OutputPluginBase() = default;
  virtual ~OutputPluginBase() = default;

  OutputPluginBase(const OutputPluginBase&) = delete;
  OutputPluginBase& operator=(const OutputPluginBase&) = delete;

  virtual void initialize(ros::NodeHandle& nh) = 0;

  virtual void execute(menoh::variable var) = 0;
};



class VGG16InputPlugin : public InputPluginBase {
 public:
  VGG16InputPlugin() = default;

  ~VGG16InputPlugin() override = default;

  void initialize(ros::NodeHandle& nh, std::vector<int32_t>& dst_dims) override;

  bool execute(menoh::variable var) override;

  void inputCallback(const sensor_msgs::ImageConstPtr& msg);

 private:
  ros::Subscriber sub_;
  sensor_msgs::ImageConstPtr latest_image_;
  std::mutex image_mutex_;
  double scale_;
  int32_t input_size_;
};

class VGG16OutputPlugin : public OutputPluginBase {
 public:
  VGG16OutputPlugin() = default;

  ~VGG16OutputPlugin() override = default;

  void initialize(ros::NodeHandle& nh) override;

  void execute(menoh::variable var) override;

 private:
  ros::Publisher  pub_;
  std::string synset_words_path_;
};

class MenohNodelet : public nodelet::Nodelet {
 public:
  MenohNodelet();

  ~MenohNodelet() override = default;

  void onInit() override;
 private:
  void timerCallback(const ros::TimerEvent& event);

  void inputCallback(const std_msgs::Float32MultiArray::ConstPtr& msg);

  std::unique_ptr<menoh::model> model_;

  std::string backend_name_;

  std::unique_ptr<InputPluginBase> input_plugin_;
  std::unique_ptr<OutputPluginBase> output_plugin_;

  std::string input_variable_name_;
  std::string output_variable_name_;

  ros::Timer timer_;

  ros::Subscriber input_sub_;
  ros::Publisher output_pub_;
};

}  // namespace menoh_ros

#endif /*MENOH_ROS_NODELET_H_*/
