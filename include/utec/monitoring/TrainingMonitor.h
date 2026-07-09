#ifndef UTEC_MONITORING_TRAINING_MONITOR_H
#define UTEC_MONITORING_TRAINING_MONITOR_H

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace utec::monitoring {

struct EpochRecord {
    int epoch = 0;
    double loss = 0.0;
    double accuracy = 0.0;
    long long duration_ms = 0;
};

class TrainingMonitor {
public:
    void on_epoch_end(int epoch, double loss, double accuracy, long long duration_ms) {
        if (!std::isfinite(loss) || !std::isfinite(accuracy)) throw std::invalid_argument("metricas invalidas");
        if (!history_.empty() && epoch <= history_.back().epoch) throw std::invalid_argument("epoca duplicada o desordenada");
        history_.push_back({epoch, loss, accuracy, duration_ms});
    }

    const std::vector<EpochRecord>& history() const { return history_; }

    std::string to_csv() const {
        std::ostringstream out;
        out << "epoch,loss,accuracy,duration_ms\n";
        for (const auto& r : history_) {
            out << r.epoch << ',' << std::setprecision(8) << r.loss << ',' << r.accuracy << ',' << r.duration_ms << '\n';
        }
        return out.str();
    }

    void write_csv(const std::string& path) const {
        std::filesystem::create_directories(std::filesystem::path(path).parent_path());
        std::ofstream out(path);
        if (!out) throw std::runtime_error("No se pudo escribir el CSV");
        out << to_csv();
    }

private:
    std::vector<EpochRecord> history_;
};

} // namespace utec::monitoring

#endif
