cmake_minimum_required(VERSION 3.27)
project(tvm_packer)
set(CMAKE_CXX_STANDARD 17)

include_directories(/workspace/tvm/include
        /workspace/tvm-wrapper/dmlc-core/include
        /workspace/tvm-wrapper/dlpack/include)

add_executable(tvm_packer main.cpp
        greedy.h
)

target_link_libraries(tvm_packer /workspace/tvm/build/libtvm.a
        /workspace/tvm/build/libbacktrace/lib/libbacktrace.a
        /workspace/tvm-wrapper/dmlc-core/libdmlc.a)
