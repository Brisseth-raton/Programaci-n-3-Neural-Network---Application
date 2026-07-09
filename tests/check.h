#ifndef CHECK_H
#define CHECK_H

#include <bits/stdc++.h>

class TestRunner {
public:
    void check(bool condition,const std::string& message,int line){
        ++total_;
        if (condition) {
            ++passed_;
            std::cout << "[PASS] " << message << '\n';
        } else {
            std::cout << "[FAIL] " << message << " (line " << line << ")\n";
        }
    }

    template<typename Func>
    void checkThrows(Func func, const std::string& message, int line){
        ++total_;
        try {
            func();
            std::cout << "[FAIL] " << message
                      << " (no exception, line " << line << ")\n";
        }
        catch (...) {
            ++passed_;
            std::cout << "[PASS] " << message << '\n';
        }
    }

    int summary() const{
        std::cout << "\n=====================================\n";
        std::cout << "Passed: " << passed_ << "/" << total_ << '\n';

        if (passed_ == total_)
            std::cout << "ALL TESTS PASSED\n";
        else
            std::cout << (total_ - passed_)
                      << " TEST(S) FAILED\n";

        std::cout << "=====================================\n";

        return passed_ == total_ ? 0 : 1;
    }

private:
    int total_ = 0;
    int passed_ = 0;
};

#define CHECK(runner, expr) \
runner.check((expr), #expr, __LINE__)

#define CHECK_THROWS(runner, expr) \
runner.checkThrows([&](){ expr; }, #expr, __LINE__)

#endif