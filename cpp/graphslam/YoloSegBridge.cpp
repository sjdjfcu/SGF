#include "YoloSegBridge.h"

#include <json11.hpp>
#include <ORUtils/Logging.h>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <poll.h>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

using namespace PSLAM;

namespace {

bool WriteAll(int fd, const void *data, size_t size) {
    const auto *bytes = static_cast<const uint8_t *>(data);
    size_t written = 0;
    while (written < size) {
        const auto result = write(fd, bytes + written, size - written);
        if (result <= 0) return false;
        written += static_cast<size_t>(result);
    }
    return true;
}

bool ReadAll(int fd, void *data, size_t size) {
    auto *bytes = static_cast<uint8_t *>(data);
    size_t read_bytes = 0;
    while (read_bytes < size) {
        const auto result = read(fd, bytes + read_bytes, size - read_bytes);
        if (result <= 0) return false;
        read_bytes += static_cast<size_t>(result);
    }
    return true;
}

void CloseFd(int *fd) {
    if (fd != nullptr && *fd >= 0) {
        close(*fd);
        *fd = -1;
    }
}

template <typename T>
void AppendValue(std::vector<uint8_t> *payload, const T &value) {
    const auto *ptr = reinterpret_cast<const uint8_t *>(&value);
    payload->insert(payload->end(), ptr, ptr + sizeof(T));
}

template <typename T>
bool ReadValue(const std::vector<uint8_t> &payload, size_t *offset, T *value) {
    if (offset == nullptr || value == nullptr) return false;
    if (*offset + sizeof(T) > payload.size()) return false;
    std::memcpy(value, payload.data() + *offset, sizeof(T));
    *offset += sizeof(T);
    return true;
}

}  // namespace

YoloSegBridge::YoloSegBridge(std::string model_path, std::string python_executable, std::vector<std::string> classes)
    : mModelPath(std::move(model_path)),
      mPythonExecutable(std::move(python_executable)),
      mClasses(std::move(classes)) {
    mReady = Start();
}

YoloSegBridge::~YoloSegBridge() {
    Stop();
}

bool YoloSegBridge::IsReady() const {
    return mReady;
}

const std::map<std::string, int> &YoloSegBridge::GetClassNameToId() const {
    return mClassNameToId;
}

std::filesystem::path YoloSegBridge::ResolveScriptPath() const {
    return std::filesystem::path(PSLAM_SOURCE_DIR) / "python" / "sgf_habitat" / "bridges" / "yolo_seg.py";
}

bool YoloSegBridge::Start() {
    const auto script_path = ResolveScriptPath();
    if (!std::filesystem::exists(script_path)) {
        SCLOG(WARNING) << "Cannot find YOLO bridge script at " << script_path.string();
        return false;
    }
    if (!std::filesystem::exists(mModelPath)) {
        SCLOG(WARNING) << "Cannot find YOLO-seg model at " << mModelPath;
        return false;
    }

    int request_pipe[2] = {-1, -1};
    int response_pipe[2] = {-1, -1};
    if (pipe(request_pipe) != 0 || pipe(response_pipe) != 0) {
        SCLOG(WARNING) << "Failed to create pipes for YOLO bridge.";
        CloseFd(&request_pipe[0]);
        CloseFd(&request_pipe[1]);
        CloseFd(&response_pipe[0]);
        CloseFd(&response_pipe[1]);
        return false;
    }

    mServerPid = fork();
    if (mServerPid == -1) {
        SCLOG(WARNING) << "Failed to fork YOLO bridge process.";
        CloseFd(&request_pipe[0]);
        CloseFd(&request_pipe[1]);
        CloseFd(&response_pipe[0]);
        CloseFd(&response_pipe[1]);
        return false;
    }

    if (mServerPid == 0) {
        dup2(request_pipe[0], STDIN_FILENO);
        dup2(response_pipe[1], STDOUT_FILENO);
        CloseFd(&request_pipe[0]);
        CloseFd(&request_pipe[1]);
        CloseFd(&response_pipe[0]);
        CloseFd(&response_pipe[1]);
        std::vector<char *> argv;
        argv.push_back(const_cast<char *>(mPythonExecutable.c_str()));
        argv.push_back(const_cast<char *>(script_path.c_str()));
        argv.push_back(const_cast<char *>("--model"));
        argv.push_back(const_cast<char *>(mModelPath.c_str()));
        for (auto &class_name : mClasses) {
            argv.push_back(const_cast<char *>("--classes"));
            argv.push_back(const_cast<char *>(class_name.c_str()));
        }
        argv.push_back(nullptr);
        execvp(mPythonExecutable.c_str(), argv.data());
        _exit(127);
    }

    mRequestFd = request_pipe[1];
    mResponseFd = response_pipe[0];
    CloseFd(&request_pipe[0]);
    CloseFd(&response_pipe[1]);

    std::vector<uint8_t> handshake_payload;
    constexpr int kHandshakeTimeoutMs = 120000;
    if (!ReadFramedPayload(&handshake_payload, kHandshakeTimeoutMs)) {
        SCLOG(WARNING) << "Timed out waiting for YOLO bridge handshake.";
        Stop();
        return false;
    }
    if (!ParseHandshake(handshake_payload)) {
        Stop();
        return false;
    }
    return true;
}

void YoloSegBridge::Stop() {
    CloseFd(&mRequestFd);

    if (mServerPid > 0) {
        constexpr int kTimeoutMs = 2000;
        constexpr int kPollMs = 50;
        bool stopped = false;
        for (int elapsed_ms = 0; elapsed_ms < kTimeoutMs; elapsed_ms += kPollMs) {
            int status = 0;
            const auto wait_result = waitpid(mServerPid, &status, WNOHANG);
            if (wait_result == mServerPid) {
                stopped = true;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
        }

        if (!stopped) {
            kill(mServerPid, SIGTERM);
            waitpid(mServerPid, nullptr, 0);
        }
    }

    CloseFd(&mResponseFd);
    mServerPid = -1;
    mReady = false;
}

bool YoloSegBridge::WriteFramedPayload(const std::vector<uint8_t> &payload) {
    if (mRequestFd < 0) return false;

    const uint64_t payload_size = payload.size();
    if (!WriteAll(mRequestFd, &payload_size, sizeof(payload_size))) return false;
    if (payload.empty()) return true;
    return WriteAll(mRequestFd, payload.data(), payload.size());
}

bool YoloSegBridge::ReadFramedPayload(std::vector<uint8_t> *payload, int timeout_ms) {
    if (payload == nullptr || mResponseFd < 0) return false;
    payload->clear();

    struct pollfd pfd {};
    pfd.fd = mResponseFd;
    pfd.events = POLLIN | POLLERR | POLLHUP;
    const auto poll_result = poll(&pfd, 1, timeout_ms);
    if (poll_result <= 0) return false;
    if ((pfd.revents & (POLLERR | POLLHUP)) != 0 && (pfd.revents & POLLIN) == 0) return false;

    uint64_t payload_size = 0;
    if (!ReadAll(mResponseFd, &payload_size, sizeof(payload_size))) return false;
    if (payload_size > (1ULL << 31)) {
        SCLOG(WARNING) << "YOLO bridge returned an unexpectedly large payload.";
        return false;
    }

    payload->resize(static_cast<size_t>(payload_size));
    if (payload->empty()) return true;
    return ReadAll(mResponseFd, payload->data(), payload->size());
}

bool YoloSegBridge::ParseHandshake(const std::vector<uint8_t> &payload) {
    const std::string content(reinterpret_cast<const char *>(payload.data()), payload.size());
    std::string error_message;
    const auto json = json11::Json::parse(content, error_message);
    if (!error_message.empty()) {
        SCLOG(WARNING) << "Failed to parse YOLO bridge handshake: " << error_message;
        return false;
    }
    if (!json["ok"].bool_value()) {
        SCLOG(WARNING) << "YOLO bridge initialization failed: " << json["error"].string_value();
        return false;
    }

    mClassNameToId.clear();
    mClassIdToName.clear();
    try {
        for (const auto &pair : json["names"].object_items()) {
            const int class_id = std::stoi(pair.first);
            const auto class_name = pair.second.string_value();
            mClassNameToId[class_name] = class_id;
            mClassIdToName[class_id] = class_name;
        }
    } catch (const std::exception &error) {
        SCLOG(WARNING) << "Invalid YOLO bridge class mapping: " << error.what();
        return false;
    }

    return true;
}

bool YoloSegBridge::ParseResponsePayload(const std::vector<uint8_t> &payload, std::vector<Detection> *detections) {
    if (detections == nullptr) return false;
    detections->clear();

    size_t offset = 0;
    uint32_t ok = 0;
    uint32_t error_length = 0;
    uint32_t detection_count = 0;
    if (!ReadValue(payload, &offset, &ok)) return false;
    if (!ReadValue(payload, &offset, &error_length)) return false;
    if (offset + error_length > payload.size()) return false;
    const std::string error_message(reinterpret_cast<const char *>(payload.data() + offset), error_length);
    offset += error_length;
    if (!ReadValue(payload, &offset, &detection_count)) return false;

    if (ok == 0) {
        SCLOG(WARNING) << "YOLO bridge inference failed: " << error_message;
        return false;
    }

    detections->reserve(detection_count);
    for (uint32_t idx = 0; idx < detection_count; ++idx) {
        Detection detection;
        int32_t class_id = -1;
        float confidence = 0.0f;
        int32_t bbox_x = 0;
        int32_t bbox_y = 0;
        int32_t bbox_w = 0;
        int32_t bbox_h = 0;
        uint32_t mask_rows = 0;
        uint32_t mask_cols = 0;
        uint32_t mask_bytes = 0;
        if (!ReadValue(payload, &offset, &class_id)) return false;
        if (!ReadValue(payload, &offset, &confidence)) return false;
        if (!ReadValue(payload, &offset, &bbox_x)) return false;
        if (!ReadValue(payload, &offset, &bbox_y)) return false;
        if (!ReadValue(payload, &offset, &bbox_w)) return false;
        if (!ReadValue(payload, &offset, &bbox_h)) return false;
        if (!ReadValue(payload, &offset, &mask_rows)) return false;
        if (!ReadValue(payload, &offset, &mask_cols)) return false;
        if (!ReadValue(payload, &offset, &mask_bytes)) return false;
        if (offset + mask_bytes > payload.size()) return false;

        detection.class_id = class_id;
        detection.class_name = mClassIdToName.count(class_id) ? mClassIdToName.at(class_id) : std::to_string(class_id);
        detection.confidence = confidence;
        detection.bbox = cv::Rect(bbox_x, bbox_y, std::max(0, bbox_w), std::max(0, bbox_h));
        if (mask_rows > 0 && mask_cols > 0 && mask_bytes == mask_rows * mask_cols) {
            detection.mask = cv::Mat(static_cast<int>(mask_rows),
                                     static_cast<int>(mask_cols),
                                     CV_8UC1,
                                     const_cast<uint8_t *>(payload.data() + offset))
                                 .clone();
        }
        offset += mask_bytes;
        detections->push_back(std::move(detection));
    }
    return true;
}

std::vector<YoloSegBridge::Detection> YoloSegBridge::Infer(const cv::Mat &image,
                                                           float conf_threshold,
                                                           float iou_threshold,
                                                           bool include_masks) {
    std::vector<Detection> detections;
    if (!mReady || image.empty()) return detections;

    cv::Mat contiguous_image;
    if (image.isContinuous()) contiguous_image = image;
    else contiguous_image = image.clone();

    const uint32_t width = static_cast<uint32_t>(contiguous_image.cols);
    const uint32_t height = static_cast<uint32_t>(contiguous_image.rows);
    const uint32_t channels = static_cast<uint32_t>(contiguous_image.channels());
    const uint32_t request_masks = include_masks ? 1U : 0U;
    const uint32_t image_bytes = static_cast<uint32_t>(contiguous_image.total() * contiguous_image.elemSize());

    std::vector<uint8_t> request_payload;
    request_payload.reserve(sizeof(width) * 5 + sizeof(float) * 2 + image_bytes);
    AppendValue(&request_payload, width);
    AppendValue(&request_payload, height);
    AppendValue(&request_payload, channels);
    AppendValue(&request_payload, conf_threshold);
    AppendValue(&request_payload, iou_threshold);
    AppendValue(&request_payload, request_masks);
    AppendValue(&request_payload, image_bytes);
    const auto *image_ptr = contiguous_image.ptr<uint8_t>(0);
    request_payload.insert(request_payload.end(), image_ptr, image_ptr + image_bytes);

    if (!WriteFramedPayload(request_payload)) {
        SCLOG(WARNING) << "Failed to send a request to the YOLO bridge.";
        Stop();
        return detections;
    }

    std::vector<uint8_t> response_payload;
    if (!ReadFramedPayload(&response_payload, 60000)) {
        SCLOG(WARNING) << "Timed out waiting for YOLO bridge response.";
        Stop();
        return detections;
    }
    if (!ParseResponsePayload(response_payload, &detections)) {
        return {};
    }
    return detections;
}
