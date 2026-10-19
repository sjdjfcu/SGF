#pragma once

#include <opencv2/core.hpp>

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <sys/types.h>

namespace PSLAM {

class YoloSegBridge {
public:
    struct Detection {
        int class_id = -1;
        std::string class_name;
        float confidence = 0.0f;
        cv::Rect bbox;
        cv::Mat mask;
    };

    YoloSegBridge(std::string model_path, std::string python_executable, std::vector<std::string> classes = {});
    ~YoloSegBridge();

    bool IsReady() const;

    std::vector<Detection> Infer(const cv::Mat &image, float conf_threshold, float iou_threshold, bool include_masks);

    const std::map<std::string, int> &GetClassNameToId() const;

private:
    bool Start();
    void Stop();
    bool ReadFramedPayload(std::vector<uint8_t> *payload, int timeout_ms);
    bool WriteFramedPayload(const std::vector<uint8_t> &payload);
    bool ParseHandshake(const std::vector<uint8_t> &payload);
    bool ParseResponsePayload(const std::vector<uint8_t> &payload, std::vector<Detection> *detections);
    std::filesystem::path ResolveScriptPath() const;

    std::string mModelPath;
    std::string mPythonExecutable;
    std::vector<std::string> mClasses;
    pid_t mServerPid = -1;
    int mRequestFd = -1;
    int mResponseFd = -1;
    bool mReady = false;
    std::map<std::string, int> mClassNameToId;
    std::map<int, std::string> mClassIdToName;
};

}  // namespace PSLAM
