# brain_ros2
Template repo for our future robot-specific projects.

## Installation
`git clone --recurse-submodules https://github.com/cornellev/brain_ros2.git`  
`cd brain_ros2 && sudo ./install.sh`

## Docker
`docker build -t cev-brain-ros2 .`

## Development

#### Formatting
If you're not using the DevContainer, please make sure to install `clang-format`, ideally version 19! This will help keep the formatting of our C++ code formatted consistently. Python code is formatted by `black`. The DevContainer will install the `Black` extension for you, which ships with `black`, but otherwise you'll need to install it in VSCode or whatever other IDE you're using.

