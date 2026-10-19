//
// Created by sc on 10/9/20.
//

#ifndef GRAPHSLAM_CONFIG_H
#define GRAPHSLAM_CONFIG_H
#include <string>
#include <vector>
#include <fstream>
#include <inseg_lib/inseg_config.h>
#include <inseg_lib/main_config.h>
#include <inseg_lib/map_config.h>
#include <inseg_lib/segmentation_config.h>
namespace PSLAM {
    struct ConfigPSLAM {
        ConfigPSLAM();
        explicit ConfigPSLAM(const std::string &path);
        ~ConfigPSLAM();
        bool use_thread;

        /// Use graph predict or not
        bool graph_predict;
        bool use_yolo_semantic;
        bool use_navigation_map;

        /// model path
        std::string pth_model;
        std::string pth_yolo_model;
        std::string python_executable;
        std::vector<std::string> yolo_classes;

        int filter_num_node;

        size_t n_pts;

        float neighbor_margin; // mm

        /// Only update the information of a node if the node has size change to N percent
        float update_thres_node_size;
        /// Update a node if its time stamp is older than n
        int update_thres_time_stamp;

        float yolo_conf_threshold;
        float yolo_iou_threshold;
        float yolo_match_iou_threshold;
        float ema_alpha_min;
        float ema_alpha_max;
        std::string yolo_match_mode;

        std::string pth;


        inseg_lib::InSegConfig inseg_config;
        inseg_lib::MainConfig main_config;
        inseg_lib::MapConfig map_config;
        inseg_lib::SegmentationConfig segmentation_config;
    };


}

#endif //GRAPHSLAM_CONFIG_H
