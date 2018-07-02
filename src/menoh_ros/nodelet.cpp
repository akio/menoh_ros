#include "menoh_ros/nodelet.h"

#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>

#include "pluginlib/class_list_macros.h"

namespace menoh_ros {


MenohNodelet::MenohNodelet() {

}

void MenohNodelet::onInit() {
  ros::NodeHandle private_nh = getPrivateNodeHandle();

  // onnx model path
  std::string onnx_model_path;
  private_nh.param<std::string>("model", onnx_model_path, "not set");

  ROS_INFO_STREAM("param = " << private_nh.resolveName("model"));
  ROS_INFO_STREAM("model = " << onnx_model_path);

  // output variable name
  private_nh.param<std::string>("input_variable_name", input_variable_name_,
                                "input");
  // output variable name
  private_nh.param<std::string>("output_variable_name", output_variable_name_,
                                "output");

  private_nh.param<std::string>("backend_name", backend_name_, "mkldnn");

  std::vector<int32_t> dims;

  if (!private_nh.getParam("input_dims", dims)) {
    ROS_ERROR_STREAM("Require parameter: " <<
                     private_nh.resolveName("input_dims"));
    return;
  }

  menoh::variable_profile_table_builder vpt_builder;
  vpt_builder.add_input_profile(input_variable_name_,
                                menoh::dtype_t::float_,
                                dims);
  vpt_builder.add_output_profile(output_variable_name_, menoh::dtype_t::float_);

  auto model_data = menoh::make_model_data_from_onnx(onnx_model_path);

  auto vpt = vpt_builder.build_variable_profile_table(model_data);

  menoh::model_builder model_builder(vpt);

  // Build model
  model_.reset(new menoh::model(model_builder.build_model(model_data,
                                                          backend_name_)));

  output_pub_ = private_nh.advertise<std_msgs::Float32MultiArray>("output", 1);
  input_sub_ = private_nh.subscribe<std_msgs::Float32MultiArray>(
       "input", 1, &MenohNodelet::inputCallback, this);
}


void MenohNodelet::inputCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
  ROS_DEBUG("Receive input");
  auto input_var = model_->get_variable(input_variable_name_);

  if (input_var.dims.size() != msg->layout.dim.size()) {
    ROS_WARN("Message dimension size didn't match: expected(%ld) != actual(%ld)",
             input_var.dims.size(), msg->layout.dim.size());
    return;
  }

  for (int i = 0; i < input_var.dims.size(); ++i) {
    if (input_var.dims[i] != msg->layout.dim[i].size) {
      ROS_WARN("Message dimension[%d] didn't match: expected(%d) != actual(%d)",
               i, input_var.dims[i], msg->layout.dim[i].size);
      return;
    }
  }

  float* input_buffer = static_cast<float*>(input_var.buffer_handle);
  std::copy(begin(msg->data) + msg->layout.data_offset,
            end(msg->data),
            input_buffer);

  model_->run();

  auto output_var = model_->get_variable(output_variable_name_);

  std_msgs::Float32MultiArray output_msg;

  size_t output_buffer_size = 1;
  for (auto dim : output_var.dims) {
    output_buffer_size *= dim;
  }
  output_msg.data.resize(output_buffer_size);
  output_msg.layout.data_offset = 0;
  output_msg.layout.dim.resize(output_var.dims.size());
  for (int i = 0; i < output_var.dims.size(); ++i) {
    output_msg.layout.dim[i].label = "dim" + std::to_string(i);
    output_msg.layout.dim[i].size = output_var.dims[i];
    output_msg.layout.dim[i].stride = 1;
    for (int j = i; j < output_var.dims.size(); ++j) {
      output_msg.layout.dim[i].stride *= output_var.dims[j];
    }
  }

  float* output_buffer = static_cast<float*>(output_var.buffer_handle);
  std::copy(output_buffer, output_buffer + output_buffer_size,
            begin(output_msg.data));

  output_pub_.publish(output_msg);
}


}  // namespace menoh_ros

PLUGINLIB_EXPORT_CLASS(menoh_ros::MenohNodelet, nodelet::Nodelet);
