#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <iomanip>
#include <cmath>
#include <limits>
#include <string>
#include <cstring>

static size_t detect_header_bytes(const std::string& path, size_t fallback = 80) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) return fallback;

    // Slurp first chunk (don’t assume exact header size)
    const size_t PROBE = 512;
    std::vector<unsigned char> buf(PROBE, 0);
    fin.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    const size_t got = static_cast<size_t>(fin.gcount());
    if (got < 64) return fallback;

    auto score_at = [&](size_t off) -> double {
        if (off + 4*16 > got) return -1e9; // not enough data to score
        // Parse 16 floats and score: finite, |v| < 10, and smooth (small deltas)
        int finite_ok = 0, range_ok = 0;
        double smooth = 0.0;
        double prev = std::numeric_limits<double>::quiet_NaN();

        for (int i = 0; i < 16; ++i) {
            size_t p = off + 4*i;
            uint32_t w =  (uint32_t)buf[p]
                        | (uint32_t)buf[p+1] << 8
                        | (uint32_t)buf[p+2] << 16
                        | (uint32_t)buf[p+3] << 24;
            float f;
            std::memcpy(&f, &w, sizeof(float)); // little-endian
            if (std::isfinite(f)) {
                finite_ok++;
                if (std::fabs(f) < 10.0) range_ok++;
                if (std::isfinite(prev)) smooth += 1.0 / (1.0 + std::fabs((double)f - prev));
                prev = f;
            } else {
                prev = std::numeric_limits<double>::quiet_NaN();
            }
        }
        // Weighted score; tuned for low-voltage analog captures
        return 3.0*finite_ok + 2.0*range_ok + smooth;
    };

    // Try likely 4-byte aligned candidates (include 48 and 80 specifically)
    std::vector<size_t> cands;
    for (size_t o = 0; o <= 128; o += 4) cands.push_back(o);

    double best_score = -1e9;
    size_t best_off = fallback;
    for (size_t off : cands) {
        double s = score_at(off);
        // prefer fallback on ties
        if (s > best_score + 1e-9 || (std::fabs(s - best_score) <= 1e-9 && off == fallback)) {
            best_score = s;
            best_off = off;
        }
    }
    return best_off;
}

int main(int argc, char* argv[]) {
    // ---- command line argument ----
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/analog_X.bin>\n";
        return 1;
    }
    const std::string path = argv[1];

    // ---- configuration ----
    const double fs       = 781250.0;              // 781.25 kS/s
    const bool useFloat32 = true;                  // true: float32, false: int16
    const double t0_s     = 0.000075122;           // absolute time of sample index 0
    const int time_sample_offset = 0;              // per-file fine nudge if ever needed
    const double start_s  = 0.0;                   // absolute start time (print window)
    const double end_s    = 0.001;                 // absolute end time
    const size_t decimate = 1;                     // 1 = every sample

    // ---- detect header length (handles ch0/ch1=80, ch2=48, etc.) ----
    const size_t headerBytes = detect_header_bytes(path, /*fallback*/80);

    // ---- open file ----
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        std::cerr << "Error: could not open " << path << "\n";
        return 1;
    }
    std::streamsize bytes = f.tellg();
    if (bytes <= static_cast<std::streamsize>(headerBytes)) {
        std::cerr << "Error: file too small or header too large. (header=" << headerBytes << ")\n";
        return 1;
    }
    f.seekg(static_cast<std::streamoff>(headerBytes)); // skip header

    std::cout << std::fixed;
    std::cout << "Time [s],Channel 0\n";

    auto clamp_and_grid = [&](size_t &i0, size_t &i1, size_t n) -> bool {
        if (i0 >= n) i0 = (n ? n - 1 : 0);
        if (i1 >  n) i1 = n;
        if (i1 <= i0) return false;
        if (decimate > 1) i0 = (i0 + decimate - 1) / decimate * decimate;
        return true;
    };

    if (useFloat32) {
        const size_t n = static_cast<size_t>((bytes - headerBytes) / 4);
        std::vector<float> data(n);
        if (!f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * 4))) {
            std::cerr << "Error: failed to read float32 samples.\n";
            return 1;
        }

        size_t i0 = (start_s <= t0_s) ? 0 : static_cast<size_t>(std::llround((start_s - t0_s) * fs));
        size_t i1 = (end_s   <= t0_s) ? 0 : static_cast<size_t>(std::llround((end_s   - t0_s) * fs));
        if (!clamp_and_grid(i0, i1, n)) {
            std::cerr << "Selected time window has no samples.\n";
            return 1;
        }

        for (size_t i = i0; i < i1; i += decimate) {
            const double t = t0_s + static_cast<double>(static_cast<long long>(i) + time_sample_offset) / fs;
            const double y = static_cast<double>(data[i]);
            std::cout << std::setprecision(9) << t << ',' << std::setprecision(3) << y << "\n";
        }
    } else {
        const size_t n = static_cast<size_t>((bytes - headerBytes) / 2);
        std::vector<int16_t> data(n);
        if (!f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(n * 2))) {
            std::cerr << "Error: failed to read int16 samples.\n";
            return 1;
        }

        size_t i0 = (start_s <= t0_s) ? 0 : static_cast<size_t>(std::llround((start_s - t0_s) * fs));
        size_t i1 = (end_s   <= t0_s) ? 0 : static_cast<size_t>(std::llround((end_s   - t0_s) * fs));
        if (!clamp_and_grid(i0, i1, n)) {
            std::cerr << "Selected time window has no samples.\n";
            return 1;
        }

        for (size_t i = i0; i < i1; i += decimate) {
            const double t = t0_s + static_cast<double>(static_cast<long long>(i) + time_sample_offset) / fs;
            const double y = static_cast<double>(data[i]) / 32768.0;
            std::cout << std::setprecision(9) << t << ',' << std::setprecision(3) << y << "\n";
        }
    }

    return 0;
}
