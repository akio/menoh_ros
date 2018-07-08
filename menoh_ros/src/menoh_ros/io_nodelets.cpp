#include "menoh_ros/io_nodelets.h"

#include "pluginlib/class_list_macros.h"

#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"
#include "std_msgs/String.h"

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

void ImageInputNodelet::onInit() {
  auto private_nh = getPrivateNodeHandle();
  // "input image width and height size"
  private_nh.param<int>("input_size", input_size_, 224);
  auto height = input_size_;
  auto width = input_size_;

  private_nh.param("scale", scale_, 1.0);
  const int batch_size = 1;
  const int channel_num = 3;

  sub_ = private_nh.subscribe("input", 1, &ImageInputNodelet::imageCallback, this);
  pub_ = private_nh.advertise<std_msgs::Float32MultiArray>("output", 1);
}

void ImageInputNodelet::imageCallback(const sensor_msgs::Image::ConstPtr& msg) {
  auto cv_image = cv_bridge::toCvShare(msg);
  auto image_mat = cv_image->image;
  auto height = input_size_;
  auto width = input_size_;

  image_mat = crop_and_resize(std::move(image_mat), cv::Size(width, height));

  std_msgs::Float32MultiArray tensor_msg;
  tensor_msg.data = reorder_to_nchw(image_mat);
  tensor_msg.layout.data_offset = 0;
  tensor_msg.layout.dim.resize(4);
  tensor_msg.layout.dim[0].label = "batch";
  tensor_msg.layout.dim[0].size = 1;
  tensor_msg.layout.dim[1].label = "channel";
  tensor_msg.layout.dim[1].size = image_mat.channels();
  tensor_msg.layout.dim[2].label = "height";
  tensor_msg.layout.dim[2].size = image_mat.cols;
  tensor_msg.layout.dim[3].label = "width";
  tensor_msg.layout.dim[3].size = image_mat.rows;
  pub_.publish(tensor_msg);
}


void CategoryOutputNodelet::onInit() {
  auto private_nh = getPrivateNodeHandle();
  private_nh.param<std::string>("category_names_path", category_names_path_, "not set");
  sub_ = private_nh.subscribe("input", 1, &CategoryOutputNodelet::resultCallback, this);
  pub_ = private_nh.advertise<std_msgs::String>("output", 1);
}


void CategoryOutputNodelet::resultCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
  // Get output
  auto categories = load_category_list(category_names_path_);
  auto top_k = 5;
  auto top_k_indices = extract_top_k_index_list(
      begin(msg->data),
      end(msg->data),
      top_k);
  ROS_INFO_STREAM("top " << top_k << " categories:");
  for(auto ki : top_k_indices) {
    ROS_INFO_STREAM("     " << ki << " " << msg->data[ki] << " "
      << categories.at(ki));
  }

  std_msgs::String result_msg;
  result_msg.data = categories.at(top_k_indices[0]);
  pub_.publish(result_msg);
}

}  // namespace menoh_ros

PLUGINLIB_EXPORT_CLASS(menoh_ros::ImageInputNodelet, nodelet::Nodelet);
PLUGINLIB_EXPORT_CLASS(menoh_ros::CategoryOutputNodelet, nodelet::Nodelet);
