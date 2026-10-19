//
// Created by sc on 11/4/20.
//
#include "config.h"
using namespace PSLAM;
ConfigPSLAM::ConfigPSLAM(const std::string &path){
    if(!path.empty()) pth=path;
    std::fstream f(pth,std::ios::in);
    if(!f.is_open())return;
}

ConfigPSLAM::~ConfigPSLAM(){}

ConfigPSLAM::ConfigPSLAM() {
    use_thread = true;

    /// Use graph predict or not
    graph_predict = true;
    use_yolo_semantic = false;
    use_navigation_map = false;

    /// model path
    pth_model = "/home/sc/research/PersistentSLAM/python/3DSSG/experiments/exp10_2_1/traced/";
    pth_model = "/home/sc/research/PersistentSLAM/python/3DSSG/experiments/exp10_4/traced/";
    pth_model = "/home/sc/research/PersistentSLAM/python/3DSSG/experiments/exp10_5/traced/";
    pth_yolo_model = PSLAM_SOURCE_DIR "/yoloe-26s-seg-pf.pt";
    python_executable = "python3";
    yolo_classes.clear();

    filter_num_node = 512;

    n_pts = 512;

    neighbor_margin=500; // mm

    /// Only update the information of a node if the node has size change to N percent
    update_thres_node_size = 0.2;
    /// Update a node if its time stamp is older than n
    update_thres_time_stamp = 50;

    yolo_conf_threshold = 0.25f;
    yolo_iou_threshold = 0.5f;
    yolo_match_iou_threshold = 0.15f;
    ema_alpha_min = 0.05f;
    ema_alpha_max = 0.60f;
    yolo_match_mode = "fast_bbox";


    pth = "./config_graph.txt";
}
