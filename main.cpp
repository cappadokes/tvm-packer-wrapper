#include <iostream>
#include "greedy.h"
#include <vector>
#include <string>
#include <sstream>
#include <utility>
#include <algorithm>
#include <chrono>

bool is_conflict(size_t start1, size_t end1, size_t start2, size_t end2)
{
    if (start2 >= end1 or start1 >= end2)
        return false;
    return true;
}

std::vector<std::vector<size_t>> read_input(const std::string &filepath)
{
    std::vector<std::vector<size_t>> result;
    std::ifstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "Error: Unable to open file " << filepath << std::endl;
        exit(1);
    }

    std::string header;
    getline(file, header);

    std::string line;
    while (getline(file, line))
    {
        std::stringstream ss(line);
        std::string value;

        getline(ss, value, ',');
        auto id = static_cast<size_t>(stoi(value));

        getline(ss, value, ',');
        auto lower = static_cast<size_t>(stoi(value));

        getline(ss, value, ',');
        auto upper = static_cast<size_t>(stoi(value));

        getline(ss, value, ',');
        auto size = static_cast<size_t>(stoi(value));

        result.push_back({id, lower, upper, size});
    }
    file.close();
    return result;
}

std::vector<std::vector<size_t>> create_conflicts(const std::vector<std::vector<size_t>> &buffers)
{
    std::vector<std::vector<size_t>> result;

    for (size_t i = 0; i < buffers.size(); ++i)
    {
        std::vector<size_t> inner_vector(1, (size_t)-1); // Initialize inner vector with size and value -1
        result.push_back(inner_vector);                  // Push the inner vector into result
    }
    for (auto &buff1 : buffers)
    {
        for (auto &buff2 : buffers)
        {
            if (buff1[0] == buff2[0])
                continue;
            if (is_conflict(buff1[1], buff1[2], buff2[1], buff2[2]))
            {
                result[buff1[0]].push_back(buff2[0]);
            }
        }
    }
    return result;
}

void save_to_csv(const std::vector<std::pair<size_t, std::string>> &output, const std::string &algorithm)
{
    const char *path = std::getenv("BASE_PATH");
    const char *name = std::getenv("TRACE_NAME");

    if (!(path && name))
    {
        std::cerr << "ERROR: One or more environment variables not set!" << std::endl;
        exit(1);
    }

    std::string path_string = std::string(path);
    std::string filename = std::string(name) + "-out.csv";

    std::string new_path = path_string + "/csv-out/";
    std::ofstream outfile_out(new_path + filename, std::ios::trunc);

    if (outfile_out.is_open())
    {
        outfile_out << "id,lower,upper,size,offset" << std::endl;
        for (auto const &line : output)
        {
            outfile_out << line.second;
        }
        outfile_out.close();
    }
    else
    {
        std::cout << "Could not open file: " << new_path << filename
                  << std::endl;
        exit(1);
    }
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <absolute_path_to_csv_file> <algorithm> <capacity>" << std::endl;
        return 1;
    }
    std::string filepath = argv[1];
    std::string algorithm = argv[2];
    std::string capacity_str = argv[3];

    size_t capacity = std::strtoull(capacity_str.c_str(), nullptr, 10);

    std::vector<std::vector<size_t>> buffers = read_input(filepath);
    std::vector<std::vector<size_t>> conflicts = create_conflicts(buffers);

    tvm::runtime::Array<tvm::tir::usmp::BufferInfo> buffer_info_arr;

    tvm::runtime::Array<tvm::PoolInfo> pool_info_arr;
    pool_info_arr.push_back(tvm::PoolInfo("default"));

    std::vector<tvm::tir::usmp::BufferInfo> buffer_info_vec;
    buffer_info_vec.reserve(buffers.size());
    for (auto buff : buffers)
    {
        buffer_info_vec.emplace_back(std::to_string(buff[0]), buff[3], pool_info_arr);
    }

    for (size_t i = 0; i < buffer_info_vec.size(); ++i)
    {
        for (size_t conflict : conflicts[i])
        {
            if (conflict == (size_t)-1)
                continue;
            buffer_info_vec[i]->conflicts.push_back(buffer_info_vec[conflict]);
        }
    }

    for (const auto &buff : buffer_info_vec)
    {
        buffer_info_arr.push_back(buff);
    }

    tvm::runtime::Map<tvm::tir::usmp::BufferInfo, tvm::tir::usmp::PoolAllocation> result;

    std::chrono::microseconds duration;
    if (algorithm == "greedy-size")
    {
        auto start = std::chrono::high_resolution_clock::now();
        result = tvm::tir::usmp::algo::GreedyBySize(buffer_info_arr);
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    else if (algorithm == "greedy-conflict")
    {
        auto start = std::chrono::high_resolution_clock::now();
        result = tvm::tir::usmp::algo::GreedyByConflicts(buffer_info_arr);
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }
    else if (algorithm == "hillclimb")
    {
        auto start = std::chrono::high_resolution_clock::now();
        result = tvm::tir::usmp::algo::HillClimb(buffer_info_arr, capacity);
        auto end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    }

    std::cout << duration;

    std::vector<std::pair<size_t, std::string>> output;
    for (auto i : result)
    {
        std::stringstream ss;
        if (i.second->byte_offset.IntValue() < 0)
        {
            std::cerr << "Negative value, exiting..." << std::endl;
            exit(1);
        }
        ss << i.first->name_hint << "," << buffers[static_cast<size_t>(std::stoi(i.first->name_hint))][1]
           << "," << buffers[static_cast<size_t>(std::stoi(i.first->name_hint))][2]
           << "," << buffers[static_cast<size_t>(std::stoi(i.first->name_hint))][3]
           << "," << i.second->byte_offset.IntValue() << std::endl;
        std::string line = ss.str();
        output.emplace_back(static_cast<size_t>(std::stoi(i.first->name_hint)), line);
    }
    std::sort(output.begin(), output.end(), [](const auto &left, const auto &right)
              { return left.first < right.first; });

    save_to_csv(output, algorithm);
    return 0;
}
