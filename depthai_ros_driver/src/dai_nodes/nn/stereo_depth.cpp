#include "depthai_ros_driver/dai_nodes/nn/stereo_depth.hpp"

#include "camera_info_manager/camera_info_manager.hpp"
#include "cv_bridge/cv_bridge.h"
#include "depthai/device/DataQueue.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/datatype/NNData.hpp"
#include "depthai/pipeline/node/ImageManip.hpp"
#include "depthai/pipeline/node/NeuralNetwork.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_ros_driver/dai_nodes/sensors/sensor_helpers.hpp"
#include "depthai_ros_driver/param_handlers/nn_param_handler.hpp"
#include "image_transport/camera_publisher.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/node.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"

namespace depthai_ros_driver
{
namespace dai_nodes
{
namespace nn
{

StereoDepth::StereoDepth(
  const std::string & daiNodeName, rclcpp::Node * node,
  std::shared_ptr<dai::Pipeline> pipeline)
: BaseNode(daiNodeName, node, pipeline)
{
  RCLCPP_DEBUG(node->get_logger(), "Creating node %s", daiNodeName.c_str());
  setNames();
  segNode = pipeline->create<dai::node::NeuralNetwork>();
  imageManipLeft = pipeline->create<dai::node::ImageManip>();
  imageManipRight = pipeline->create<dai::node::ImageManip>();

  ph = std::make_unique<param_handlers::NNParamHandler>(node, daiNodeName);
  ph->declareParams(segNode, imageManipLeft);
  ph->declareParams(segNode, imageManipRight);

  imageManipLeft->out.link(segNode->inputs["left"]);
  imageManipRight->out.link(segNode->inputs["right"]);

  setXinXout(pipeline);
}

StereoDepth::~StereoDepth() = default;

void StereoDepth::setNames()
{
  nnQName = getName() + "_nn";
}

void StereoDepth::setXinXout(std::shared_ptr<dai::Pipeline> pipeline)
{
  xoutNN = pipeline->create<dai::node::XLinkOut>();
  xoutNN->setStreamName(nnQName);
  segNode->out.link(xoutNN->input);
}

void StereoDepth::setupQueues(std::shared_ptr<dai::Device> device)
{
  nnQ = device->getOutputQueue(nnQName, ph->getParam<int>("i_max_q_size"), false);
  infoManager = std::make_shared<camera_info_manager::CameraInfoManager>(
    getROSNode()->create_sub_node(
      std::string(
        getROSNode()->get_name()) + "/" + getName()).get(), "/" + getName());
  infoManager->setCameraInfo(
    sensor_helpers::getCalibInfo(
      getROSNode()->get_logger(),
      *imageConverter,
      device,
      dai::CameraBoardSocket::CAM_A,
      imageManipLeft->initialConfig.getResizeWidth(),
      imageManipLeft->initialConfig.getResizeHeight()));
  nnPub = image_transport::create_camera_publisher(getROSNode(), "stereo_depth/image_raw");
  nnQ->addCallback(
    std::bind(
      &StereoDepth::stereoDepthCB, this, std::placeholders::_1,
      std::placeholders::_2));
}

void StereoDepth::closeQueues()
{
  nnQ->close();
}

void StereoDepth::stereoDepthCB(
  const std::string & /*name*/,
  const std::shared_ptr<dai::ADatatype> & data)
{
  std::shared_ptr<dai::NNData> nnData = std::static_pointer_cast<dai::NNData>(data);
  std::vector<float> output = nnData->getFirstLayerFp16();

  // Drop half of the data
  std::vector<float> imageData(output.begin(), output.begin() + 160 * 240);

  cv::Mat disparity = cv::Mat(160, 240, CV_32FC1, imageData.data());

  // convert to depth
  cv::Mat depth = cv::Mat(160, 240, CV_32FC1);
  depth = 168.50592041015625 * 0.075 * 1000.0 / disparity;

  cv::Mat image = cv::Mat(160, 240, CV_16UC1);
  depth.convertTo(image, CV_16UC1);

  // filter out invalid depth
  cv::Mat mask = cv::Mat(160, 240, CV_8UC1);
  cv::inRange(depth, 0, 3000, mask);

  image.setTo(0, mask == 0);

  // convert to ros msg
  sensor_msgs::msg::Image img_msg;

  // get camera info from camera info manager
  nnInfo = sensor_msgs::msg::CameraInfo();
  nnInfo.header.frame_id = std::string(getROSNode()->get_name()) + "_rgb_camera_optical_frame";
  nnInfo.height = depth.rows;
  nnInfo.width = depth.cols;
  nnInfo.distortion_model = "rational_polynomial";
  nnInfo.header.stamp = getROSNode()->get_clock()->now();

  nnInfo.d = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  nnInfo.k =
  {168.50592041015625, 0.0, 122.49176788330078, 0.0, 168.50592041015625, 80.1346435546875,
    0.0, 0.0, 1.0};
  nnInfo.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1};
  nnInfo.p =
  {168.50592041015625, 0.0, 122.49176788330078, 0.0, 0.0, 168.50592041015625,
    80.1346435546875, 0.0, 0.0, 0.0, 1.0, 0.0};

  std_msgs::msg::Header header;
  header.stamp = nnInfo.header.stamp;
  header.frame_id = std::string(getROSNode()->get_name()) + "_rgb_camera_optical_frame";
  nnInfo.header = header;


  cv_bridge::CvImage(
    header, sensor_msgs::image_encodings::TYPE_16UC1,
    image).toImageMsg(img_msg);

  img_msg.step = img_msg.width * sizeof(uint16_t);


  nnPub.publish(img_msg, nnInfo);

}

void StereoDepth::link(dai::Node::Input in, int /*linkType*/)
{
  segNode->out.link(in);
}

dai::Node::Input StereoDepth::getInput(int linkType)
{
  if (linkType == 0) {
    return imageManipLeft->inputImage;
    //return segNode->inputs["left"];
  } else {
    return imageManipRight->inputImage;
    //return segNode->inputs["right"];
  }
}

void StereoDepth::updateParams(const std::vector<rclcpp::Parameter> & params)
{
  ph->setRuntimeParams(params);
}
}  // namespace nn
}  // namespace dai_nodes
}  // namespace depthai_ros_driver
