FROM ros:humble
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    build-essential \
    cmake \
    libeigen3-dev \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install mujoco
RUN mkdir -p /root/unitree_ws/src
RUN git clone https://github.com/unitreerobotics/unitree_ros2.git \
 /root/tmp_repo
RUN cp -r /root/tmp_repo/cyclonedds_ws/src/* /root/unitree_ws/src/
RUN rm -rf /root/tmp_repo
WORKDIR /root/unitree_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"
RUN mkdir -p /root/safety_ws/src
WORKDIR /root/safety_ws/src
COPY . .
WORKDIR /root/safety_ws
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-select safety_block"
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "source /root/unitree_ws/install/setup.bash" >> ~/.bashrc
RUN echo "source /root/safety_ws/install/setup.bash" >> ~/.bashrc
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /root/unitree_ws/install/setup.bash && source /root/safety_ws/install/setup.bash && ros2 run safety_block safety_node --ros-args -p go1_xml_path:=/root/safety_ws/src/launch/go1.xml"]
