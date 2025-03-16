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
#include "cost_map/gaussian_conv.h"
#include "cost_map/global.h"
#include "cost_map/nearest.h"

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

        global_path_sub_ =
            this->create_subscription<nav_msgs::msg::Path>("/offtrack_server/global_path", 1,
                std::bind(&PlannerNode::global_path_callback, this, std::placeholders::_1));

        // Publishers
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry/truth", 1);
        local_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("planner/local_path", 1);

        planning_timer_ =
            this->create_wall_timer(std::chrono::milliseconds(int(planning_dt * 1000)),
                std::bind(&PlannerNode::local_plan, this));

        update_timer_ = this->create_wall_timer(std::chrono::milliseconds(int(update_dt * 1000)),
            std::bind(&PlannerNode::odom_publish, this));

        cost_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("planner/cost_map",
            qos);

        // DEBUG
        first_target_pub_ =
            this->create_publisher<nav_msgs::msg::Odometry>("planner/debug/first_target", 1);
        second_target_pub_ =
            this->create_publisher<nav_msgs::msg::Odometry>("planner/debug/second_target", 1);
    }

private:
    void global_path_callback(nav_msgs::msg::Path::SharedPtr msg) {
        Trajectory traj = Trajectory();
        for (geometry_msgs::msg::PoseStamped pose: msg->poses) {
            traj.waypoints.push_back(State(pose.pose.position.x, pose.pose.position.y, 0, 0, 0));
        }

        global_path = traj;
        global_path_initialized = true;

        current_state = global_path.waypoints[0];
        current_state.pose.theta = -M_PI / 2.0;

        if (global_path.waypoints.size() > 1) {
            first_target = global_path.waypoints[1];
        } else {
            first_target = current_state;
        }

        if (global_path.waypoints.size() > 2) {
            second_target = global_path.waypoints[2];
        } else {
            second_target = first_target;
        }

        RCLCPP_INFO(this->get_logger(), "Received global path.");
    }

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
        RCLCPP_INFO(get_logger(), "Received regular map.");
    }

    void costmap_callback(nav_msgs::msg::OccupancyGrid::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "Received cost map.");
        this->costmap = SimpleCostMap(map_to_grid(msg)).toCostmap();
        this->costmap_initialized = true;

        RCLCPP_INFO(this->get_logger(), "Received cost map.");
    }

    nav_msgs::msg::Odometry state_to_odom(State state) {
        nav_msgs::msg::Odometry odom;
        odom.header.frame_id = "map";
        odom.header.stamp = this->now();
        odom.child_frame_id = "debug";
        odom.pose.pose.position.x = state.pose.x;
        odom.pose.pose.position.y = state.pose.y;
        odom.pose.pose.position.z = 0.0;

        Eigen::AngleAxisd angle_axis(state.pose.theta, Eigen::Vector3d::UnitZ());
        Eigen::Quaterniond quat(angle_axis);
        odom.pose.pose.orientation = tf2::toMsg(quat);

        return odom;
    }

    void odom_publish() {
        if (!global_path_initialized) {
            RCLCPP_DEBUG(this->get_logger(), "Waiting for path.");
            return;
        }

        // Update and publish odometry
        RCLCPP_DEBUG(this->get_logger(), "Publishing odometry.");

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

        current_state = current_state.update(current_input, update_dt, dimensions,
            full_constraints);

        // Update targets
        if (current_global_waypoint < global_path.waypoints.size()) {
            // RCLCPP_INFO(this->get_logger(), "A");
            // RCLCPP_INFO(this->get_logger(), "%f",
            //     current_state.pose.distance_to(first_target.pose));
            // Progress in path if reached first waypoint
            if (current_state.pose.distance_to(first_target.pose) < 15.0) {
                current_global_waypoint += 1;
                if (current_global_waypoint < global_path.waypoints.size()) {
                    first_target = global_path.waypoints[current_global_waypoint];
                    // RCLCPP_INFO(this->get_logger(), "progressing.");
                }
                first_target_pub_->publish(state_to_odom(first_target));

                if (current_global_waypoint + 1 < global_path.waypoints.size()) {
                    second_target = global_path.waypoints[current_global_waypoint + 1];
                } else {
                    second_target = first_target;
                }
                second_target_pub_->publish(state_to_odom(second_target));
            }
        }
    }

    bool hits_obstacle(Trajectory traj) {
        for (int i = 0; i < traj.waypoints.size(); i++) {
            if (costmap->cost(traj.waypoints[i]) > 10) {
                return true;
            }
        }

        return false;
    }

    void local_plan() {
        if (!map_initialized || !costmap_initialized || !global_path_initialized) {
            RCLCPP_DEBUG(this->get_logger(), "Waiting for map and target.");
            return;
        }

        RCLCPP_DEBUG(this->get_logger(), "Planning path.");

        float dist_to_dest = current_state.pose.distance_to(first_target.pose);
        if (dist_to_dest < 15.0) {
            RCLCPP_INFO(this->get_logger(), "Reached target");
            current_state.vel = 0;
            return;
        }

        Trajectory waypoints;
        Trajectory initial_guess;
        waypoints.waypoints.push_back(first_target);

        Trajectory path = local_planner->plan_path(map_grid, current_state, second_target,
            waypoints, initial_guess, costmap);

        if (hits_obstacle(path)) {
            RCLCPP_INFO(this->get_logger(), "Hit obstacle, discarding plan.");

            if (current_trajectory_init) {
                if (position_in_current_trajectory + 2 < current_trajectory.waypoints.size()) {
                    position_in_current_trajectory += 1;
                    current_input = Input(
                        (current_trajectory.waypoints[position_in_current_trajectory + 1].tau
                            - current_trajectory.waypoints[position_in_current_trajectory].tau)
                            / planning_dt,
                        (current_trajectory.waypoints[position_in_current_trajectory + 1].vel
                            - current_trajectory.waypoints[position_in_current_trajectory].vel)
                            / planning_dt);
                }
            }
            return;
        }

        // Update current input for odom updater
        current_input = Input((path.waypoints[1].tau - path.waypoints[0].tau) / planning_dt,
            (path.waypoints[1].vel - path.waypoints[0].vel) / planning_dt);

        position_in_current_trajectory = 0;
        current_trajectory = path;
        current_trajectory_init = true;

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
    float planning_dt = 0.2;
    float update_dt = .05;

    Input current_input = Input(0, 0);

    // Map, Costmap
    Grid map_grid;
    std::shared_ptr<cost_map::CostMap> map;
    std::shared_ptr<cost_map::CostMap> costmap;
    bool map_initialized = false;
    bool costmap_initialized = false;

    // States, Targets
    State current_state;

    State first_target;
    State second_target;

    Trajectory global_path = Trajectory();

    bool global_path_initialized = false;
    int current_global_waypoint = 0;

    // Planner
    std::shared_ptr<local_planner::MPC> local_planner;

    Dimensions dimensions = Dimensions{1, 1, 1};
    Constraints full_constraints = Constraints{
        {-1000, 1000},  // x
        {-1000, 1000},  // y
        {-.34, .34},    // tau
        {0.0, 10.0},    // vels
        {-1.0, 2.5},    // accel
        {-.10, .10}     // dtau
    };

    Trajectory current_trajectory;
    bool current_trajectory_init = false;
    int position_in_current_trajectory = 0;

    // ROS
    rclcpp::TimerBase::SharedPtr planning_timer_;
    rclcpp::TimerBase::SharedPtr update_timer_;

    // Subscribers
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr map_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr costmap_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr global_path_sub_;

    // Publishers
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr local_path_pub_;

    tf2_ros::TransformBroadcaster tf_broadcast_;

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr cost_map_pub_;

    // DEBUG
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr first_target_pub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr second_target_pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PlannerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}