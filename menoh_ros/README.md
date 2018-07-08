Menoh ROS Package
==================================================

ROS inteface of [Menoh](https://github.com/pfnet-research/menoh) library by @pfnet-research.

The MenohNodelet loads [ONNX](https://onnx.ai/) model and export input and output as ROS topics.

Thanks to the power of Menoh, it can run neural network models efficiently without GPGPUs.


Installation
--------------------------------------------------

### Catkin tools

    $ mkdir -p catkin_ws/src
    $ cd catkin_ws
    $ catkin init
    $ cd src
    $ git clone https://github.com/akio/menoh_ros.git
    $ catkin build

### APT Package

In preparation now.


Architecture 
--------------------------------------------------

Following diagram depicts the architecture of `MenohNodelet` pipeline. 

            |
            | Any Input
            V
    +-----------------+
    | InputNode(let)  |
    +-----------------+
            |
            | std_msgs/Float32MultiArray
            V
    +-----------------+
    | MenohNodelet    |<--- ONNX Model
    +-----------------+
            |
            | std_msgs/Float32MultiArray
            V
    +-----------------+
    | OutputNode(let) |
    +-----------------+
            |
            | Any Output
            V

Provided Nodelets
----------------------------------------------------

### MenohNodelet

`MenohNodelet` is a core nodelet of this package.
This nodelet loads ONNX model file and export them as `std_msgs/Float32MultiArray` topics.
When it receives a input message, it loads the message into the neural network model.
After the neural network computes the output, the nodelet translate the output into a ROS message and publish it.

### ImageInputNodelet

This nodelet subscribes `sensor_msgs/Image` and converts it to `std_msgs/Float32MultiArray` and publishes it to `MenohNodelet`.

### CategoryOutputNodelet

This nodelet subscribes `std_msgs/Float32MultiArray` and lodas category label data from a textfile.
When it receives a message, it compute softmax of the message and publish a corresponding label line as a `std_msgs/String`.


Example
-------------------------------------------------

### ROS computation graph version of VGG16 example in Menoh 

    $ python scripts/retrieve_data.py
    $ roslaunch launch/vgg16.launch

See `launch/vgg16.launch` as an example.


License
-------------------------------------------------
This package is available under terms of the [MIT License](https://opensource.org/licenses/MIT).
