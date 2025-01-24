cmake_minimum_required(VERSION 3.27)
project(tvm_packer)
set(CMAKE_CXX_STANDARD 17)

include_directories(/home/workspace/tvm/include
        /home/workspace/tvm-wrapper/dmlc-core/include
        /home/workspace/tvm-wrapper/dlpack/include)

add_executable(tvm_packer main.cpp
        greedy.h
)

target_link_libraries(tvm_packer /home/workspace/tvm/build/libtvm.a
        /home/workspace/tvm/build/libbacktrace/lib/libbacktrace.a
        /home/workspace/tvm-wrapper/dmlc-core/libdmlc.a)
