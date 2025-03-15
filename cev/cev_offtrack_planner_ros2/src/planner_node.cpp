#include <iostream>
// ROS
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include "tf2/utils.h"
#include "tf2_eigen/tf2_eigen.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/transform_broadcaster.h"

// CEV
#include <cev_msgs/msg/trajectory.hpp>
#include "local_planning/mpc.h"
#include "pose.h"
#include "cost_map/simple.h"

using namespace cev_planner;

class PlannerNode : public rclcpp::Node {
public:
    PlannerNode(): Node("planner_node"), tf_broadcast_(this) {
        RCLCPP_INFO(this->get_logger(), "Initializing planner node");

        // Init Planner
        local_planner = std::make_shared<local_planner::MPC>(dimensions, full_constraints);

        // Subscribers
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).transient_local();
        map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("/offtrack_server/map",
            qos, std::bind(&PlannerNode::map_callback, this, std::placeholders::_1));

        costmap_sub_ =
            this->create_subscription<nav_msgs::msg::OccupancyGrid>("/offtrack_server/costmap", qos,
                std::bind(&PlannerNode::costmap_callback, this, std::placeholders::_1));

        target_rviz_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>("/goal_pose",
            1, std::bind(&PlannerNode::target_callback, this, std::placeholders::_1));

        // Publishers
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/truth", 1);
        local_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("planner/local_path", 1);

        planning_timer_ =
            this->create_wall_timer(std::chrono::milliseconds(int(planning_dt * 1000)),
                std::bind(&PlannerNode::local_plan, this));

        update_timer_ = this->create_wall_timer(std::chrono::milliseconds(int(update_dt * 1000)),
            std::bind(&PlannerNode::odom_publish, this));

        cost_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("planner/cost_map", 1);
    }

private:
    Grid map_to_grid(nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        Grid grid;

        grid.data = Eigen::MatrixXf::Zero(msg->info.width, msg->info.height);
        grid.origin = Pose(msg->info.origin.position.x, msg->info.origin.position.y);
        grid.resolution = msg->info.resolution;

        for (int i = 0; i < msg->info.width; i++) {
            for (int j = 0; j < msg->info.height; j++) {
                grid.data(i, j) = msg->data[j * msg->info.width + i];
            }
        }

        return grid;
    }

    void map_callback(nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        map_grid = map_to_grid(msg);
        this->map = SimpleCostMap(map_grid).toCostmap();
        this->map_initialized = true;
    }

    void costmap_callback(nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received cost map.");
        this->costmap = SimpleCostMap(map_to_grid(msg)).toCostmap();
        this->costmap_initialized = true;

        Grid grid = Grid();
        grid.origin = Pose{msg->info.origin.position.x, msg->info.origin.position.y, 0};
        grid.resolution = msg->info.resolution;

        grid.data = Eigen::MatrixXf(msg->info.width, msg->info.height);

        double avg = 0.0;
        for (int i = 0; i < msg->info.width; i++) {
            for (int j = 0; j < msg->info.height; j++) {
                avg += msg->data[j * msg->info.width + i];
            }
        }
        avg /= grid.data.rows() * grid.data.cols();
        RCLCPP_INFO(this->get_logger(), "Cost: %f", avg);

        RCLCPP_INFO(this->get_logger(), "Generated cost maps.");

        // Convert local plan cost map to an occupancy grid message and publish
        nav_msgs::msg::OccupancyGrid cost_map_msg;
        cost_map_msg.header.stamp = this->now();
        cost_map_msg.header.frame_id = "map";
        cost_map_msg.info.resolution = grid.resolution;
        cost_map_msg.info.width = grid.data.cols();
        cost_map_msg.info.height = grid.data.rows();
        cost_map_msg.info.origin.position.x = grid.origin.x;
        cost_map_msg.info.origin.position.y = grid.origin.y;
        cost_map_msg.info.origin.position.z = 0;
        cost_map_msg.info.origin.orientation.x = 0;
        cost_map_msg.info.origin.orientation.y = 0;
        cost_map_msg.info.origin.orientation.z = 0;
        cost_map_msg.info.origin.orientation.w = 1;

        cost_map_msg.data.clear();

        for (int i = 0; i < grid.data.rows(); i++) {
            for (int j = 0; j < grid.data.cols(); j++) {
                cost_map_msg.data.push_back((int)((costmap->debug_(i, j)) * 100));
            }
        }
        nav_msgs::msg::OccupancyGrid mirrored_msg;
        mirrored_msg.header.stamp = this->now();
        mirrored_msg.header.frame_id = "map";
        mirrored_msg.info.resolution = grid.resolution;
        mirrored_msg.info.width = grid.data.rows();
        mirrored_msg.info.height = grid.data.cols();
        mirrored_msg.info.origin.position.z = 0;
        mirrored_msg.info.origin.orientation.x = 0;
        mirrored_msg.info.origin.orientation.y = 0;
        mirrored_msg.info.origin.orientation.z = 0;
        mirrored_msg.info.origin.orientation.w = 1;

        mirrored_msg.data.clear();

        for (int i = 0; i < grid.data.rows(); i++) {
            for (int j = 0; j < grid.data.cols(); j++) {
                mirrored_msg.data.push_back(0);
            }
        }

        for (int y = 0; y < cost_map_msg.info.height; ++y) {
            for (int x = 0; x < cost_map_msg.info.width; ++x) {
                int new_x = y;
                int new_y = x;
                mirrored_msg.data[new_y * cost_map_msg.info.height + new_x] =
                    cost_map_msg.data[y * cost_map_msg.info.width + x];
            }
        }

        // Swap the origin x and y
        mirrored_msg.info.origin.position.x = cost_map_msg.info.origin.position.x;
        mirrored_msg.info.origin.position.y = cost_map_msg.info.origin.position.y;

        RCLCPP_INFO(this->get_logger(), "Published cost map.");
        cost_map_pub_->publish(mirrored_msg);
    }

    void target_callback(geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received target.");
        target_state = State(msg->pose.position.x, msg->pose.position.y, 0, 0, 0);
        target_initialized = true;
    }

    void odom_publish() {
        // Update and publish odometry
        RCLCPP_DEBUG(this->get_logger(), "Publishing odometry.");

        current_state = current_state.update(current_input, update_dt, dimensions,
            full_constraints);

        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = "map";
        odom.header.stamp = this->now();
        odom.child_frame_id = "base_link";
        odom.pose.pose.position.x = current_state.pose.x;
        odom.pose.pose.position.y = current_state.pose.y;
        odom.pose.pose.position.z = 0.0;

        geometry_msgs::msg::TransformStamped tf_stamped;
        tf_stamped.header = odom.header;
        tf_stamped.child_frame_id = odom.child_frame_id;
        tf_stamped.transform.translation.x = odom.pose.pose.position.x;
        tf_stamped.transform.translation.y = odom.pose.pose.position.y;
        tf_stamped.transform.translation.z = odom.pose.pose.position.z;
        tf_stamped.transform.rotation = odom.pose.pose.orientation;
        tf_broadcast_.sendTransform(tf_stamped);

        Eigen::AngleAxisd angle_axis(current_state.pose.theta, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond quat(angle_axis);
        odom.pose.pose.orientation = tf2::toMsg(quat);

        odom_pub_->publish(odom);
    }

    void local_plan() {
        if (!map_initialized || !costmap_initialized || !target_initialized) {
            RCLCPP_DEBUG(this->get_logger(), "Waiting for map and target.");
            return;
        }

        RCLCPP_DEBUG(this->get_logger(), "Planning path.");

        float dist_to_dest = current_state.pose.distance_to(target_state.pose);
        if (dist_to_dest < 0.3) {
            RCLCPP_INFO(this->get_logger(), "Reached target");
            return;
        }

        Trajectory waypoints;
        Trajectory initial_guess;
        waypoints.waypoints.push_back(target_state);

        Trajectory path = local_planner->plan_path(map_grid, current_state, target_state, waypoints,
            initial_guess, costmap);

        current_input = Input((path.waypoints[1].tau - path.waypoints[0].tau) / planning_dt,
            (path.waypoints[1].vel - path.waypoints[0].vel) / planning_dt);

        RCLCPP_DEBUG(this->get_logger(), "Publishing local path.");

        nav_msgs::msg::Path nav_path;
        nav_path.header.stamp = this->now();
        nav_path.header.frame_id = "map";
        nav_path.poses.clear();

        for (State waypoint: path.waypoints) {
            geometry_msgs::msg::PoseStamped pose;
            pose.pose.position.x = waypoint.pose.x;
            pose.pose.position.y = waypoint.pose.y;
            pose.pose.position.z = 0;
            pose.pose.orientation = tf2::toMsg(tf2::Quaternion(tf2::Vector3(0, 0, 1),
                waypoint.pose.theta));
            nav_path.poses.push_back(pose);
        }

        local_path_pub_->publish(nav_path);
    }

    // SETTINGS
    float planning_dt = 0.4;
    float update_dt = .05;

    Input current_input = Input(0, 0);

    // Map, Costmap
    Grid map_grid;
    std::shared_ptr<cost_map::CostMap> map;
    std::shared_ptr<cost_map::CostMap> costmap;
    bool map_initialized = false;
    bool costmap_initialized = false;

    // States, Targets
    State current_state = State(13.0, 298.913, M_PI / 2.0, 0.0, 0.0);
    State target_state = State();

    bool target_initialized = false;

    // Planner
    std::shared_ptr<local_planner::MPC> local_planner;

    Dimensions dimensions = Dimensions{1, 1, 1};
    Constraints full_constraints = Constraints{
        {-1000, 1000},  // x
        {-1000, 1000},  // y
        {-.34, .34},    // tau
        {0.0, 10.0},    // vels
        {-7.5, 5.0},    // accel
        {-.20, .20}     // dtau
    };

    // ROS
    rclcpp::TimerBase::SharedPtr planning_timer_;
    rclcpp::TimerBase::SharedPtr update_timer_;

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr target_rviz_sub_;

    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub_;

    tf2_ros::TransformBroadcaster tf_broadcast_;

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_map_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}