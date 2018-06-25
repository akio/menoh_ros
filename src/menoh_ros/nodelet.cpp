#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <string>
#include <vector>
#include <mutex>

#include "pluginlib/class_list_macros.h"

#include "opencv2/opencv.hpp"

#include "nodelet/nodelet.h"
#include "cv_bridge/cv_bridge.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "std_msgs/String.h"

#include "menoh/menoh.hpp"

namespace menoh_ros {

cv::Mat crop_and_resize(cv::Mat mat, cv::Size const& size) {
    auto short_edge = std::min(mat.size().width, mat.size().height);
    cv::Rect roi;
    roi.x = (mat.size().width - short_edge) / 2;
    roi.y = (mat.size().height - short_edge) / 2;
    roi.width = roi.height = short_edge;
    cv::Mat cropped = mat(roi);
    cv::Mat resized;
    cv::resize(cropped, resized, size);
    return resized;
}

std::vector<float> reorder_to_nchw(cv::Mat const& mat) {
    assert(mat.channels() == 3);
    std::vector<float> data(mat.channels() * mat.rows * mat.cols);
    for(int y = 0; y < mat.rows; ++y) {
        for(int x = 0; x < mat.cols; ++x) {
            // INFO cv::imread loads image BGR
            for(int c = 0; c < mat.channels(); ++c) {
                data[c * (mat.rows * mat.cols) + y * mat.cols + x] =
                  static_cast<float>(
                    mat.data[y * mat.step + x * mat.elemSize() + c]);
            }
        }
    }
    return data;
}

template <typename InIter>
std::vector<typename std::iterator_traits<InIter>::difference_type>
extract_top_k_index_list(
  InIter first, InIter last,
  typename std::iterator_traits<InIter>::difference_type k) {
    using diff_t = typename std::iterator_traits<InIter>::difference_type;
    std::priority_queue<
      std::pair<typename std::iterator_traits<InIter>::value_type, diff_t>>
      q;
    for(diff_t i = 0; first != last; ++first, ++i) {
        q.push({*first, i});
    }
    std::vector<diff_t> indices;
    for(diff_t i = 0; i < k; ++i) {
        indices.push_back(q.top().second);
        q.pop();
    }
    return indices;
}



std::vector<std::string> load_category_list(std::string const& synset_words_path) {
    std::ifstream ifs(synset_words_path);
    if(!ifs) {
        throw std::runtime_error("File open error: " + synset_words_path);
    }
    std::vector<std::string> categories;
    std::string line;
    while(std::getline(ifs, line)) {
        categories.push_back(std::move(line));
    }
    return categories;
}


class InputPluginBase {
 public:
  InputPluginBase() = default;
  virtual ~InputPluginBase() = default;

  InputPluginBase(const InputPluginBase&) = delete;
  InputPluginBase& operator=(const InputPluginBase&) = delete;

  virtual void initialize(ros::NodeHandle& nh,
                          std::vector<int32_t>& dst_dims) = 0;

  virtual void execute(menoh::variable var) = 0;
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

  void initialize(ros::NodeHandle& nh, std::vector<int32_t>& dst_dims) override;

  void execute(menoh::variable var) override;

  void inputCallback(const sensor_msgs::ImageConstPtr& msg);

 private:
  ros::Subscriber sub_;
  sensor_msgs::ImageConstPtr latest_image_;
  std::mutex image_mutex_;
  double scale_;
  int32_t input_size_;
};

void VGG16InputPlugin::initialize(ros::NodeHandle& nh,
                                  std::vector<int32_t>& dst_dims) {
  // "input image width and height size"
  nh.param<int>("input_size", input_size_);
  auto height = input_size_;
  auto width = input_size_;

  nh.param("scale", scale_, 1.0);
  const int batch_size = 1;
  const int channel_num = 3;

  dst_dims = {batch_size, channel_num, height, width};

  sub_ = nh.subscribe("input", 1, &VGG16InputPlugin::inputCallback, this);
}

void VGG16InputPlugin::execute(menoh::variable var) {
  std::lock_guard<std::mutex> lock(image_mutex_);
  auto cv_image = cv_bridge::toCvShare(latest_image_);
  auto image_mat = cv_image->image;
  auto height = input_size_;
  auto width = input_size_;

  image_mat = crop_and_resize(std::move(image_mat), cv::Size(width, height));
  auto image_data = reorder_to_nchw(image_mat);

  float* input_buff = static_cast<float*>(var.buffer_handle);
  std::copy(begin(image_data), end(image_data), input_buff);
}

void VGG16InputPlugin::inputCallback(const sensor_msgs::ImageConstPtr& msg) {
  std::lock_guard<std::mutex> lock(image_mutex_);
  latest_image_ = msg;
}

class VGG16OutputPlugin : public OutputPluginBase {
 public:
  VGG16OutputPlugin() = default;

  void initialize(ros::NodeHandle& nh) override;

  void execute(menoh::variable var) override;

 private:
  ros::Publisher  pub_;
};

void VGG16OutputPlugin::initialize(ros::NodeHandle& nh) {
  pub_ = nh.advertise<std_msgs::String>("output", 1);
}

void VGG16OutputPlugin::execute(menoh::variable var) {
  float* softmax_output_buff = static_cast<float*>(var.buffer_handle);

  // Get output
  auto top_k = 5;
  auto top_k_indices = extract_top_k_index_list(
      softmax_output_buff,
      softmax_output_buff + var.dims.at(1),
      top_k);

  ROS_INFO_STREAM("top " << top_k << " categories are");
  for(auto ki : top_k_indices) {
    std::cout << ki << " " << *(softmax_output_buff + ki) << std::endl;
  }

  std_msgs::String msg;
  pub_.publish(msg);
}

class MenohNodelet : public nodelet::Nodelet {
 public:
  MenohNodelet();

  void onInit() override;
 private:
  void timerCallback(const ros::TimerEvent& event);

  std::unique_ptr<menoh::model> model_;

  std::string backend_name_;

  std::unique_ptr<InputPluginBase> input_plugin_;
  std::unique_ptr<OutputPluginBase> output_plugin_;

  std::string input_variable_name_;
  std::string output_variable_name_;

  ros::Timer timer_;
};

MenohNodelet::MenohNodelet() {

}

void MenohNodelet::onInit() {
  auto nh = getNodeHandle();

  // onnx model path
  std::string onnx_model_path;
  nh.param<std::string>("model", onnx_model_path);
  // output variable name
  nh.param<std::string>("input_variable_name", input_variable_name_);
  // output variable name
  nh.param<std::string>("output_variable_name", output_variable_name_);

  nh.param<std::string>("backend_name", backend_name_, "mkldnn");

  std::vector<int32_t> dims;
  input_plugin_.reset(new VGG16InputPlugin());
  input_plugin_->initialize(nh, dims);

  output_plugin_.reset(new VGG16OutputPlugin());
  output_plugin_->initialize(nh);

  menoh::variable_profile_table_builder vpt_builder;
  vpt_builder.add_input_profile(input_variable_name_,
                                menoh::dtype_t::float_,
                                dims);
  vpt_builder.add_output_profile(output_variable_name_, menoh::dtype_t::float_);

  auto model_data = menoh::make_model_data_from_onnx(onnx_model_path);

  auto vpt = vpt_builder.build_variable_profile_table(model_data);

  menoh::model_builder model_builder(vpt);

  // Build model
  model_.reset(new menoh::model(model_builder.build_model(model_data, backend_name_)));

  timer_ = nh.createTimer(ros::Duration(), &MenohNodelet::timerCallback, this);
}

void MenohNodelet::timerCallback(const ros::TimerEvent& event) {
  // Get buffer pointer of output
  auto input_var = model_->get_variable(input_variable_name_);
  input_plugin_->execute(input_var);

  // Run inference
  model_->run();

  auto output_var = model_->get_variable(output_variable_name_);
  output_plugin_->execute(output_var);
}


}  // namespace menoh_ros

PLUGINLIB_EXPORT_CLASS(menoh_ros::MenohNodelet, nodelet::Nodelet);
