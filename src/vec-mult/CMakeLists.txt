cmake_minimum_required(VERSION 3.18)

project(vec-mult LANGUAGES Swift)

set(SWIFT_TARGET SwiftMetalMult)
add_executable(${SWIFT_TARGET} vec-mult.swift)

# Link Metal and MetalPerformanceShaders frameworks
target_link_libraries(${SWIFT_TARGET} PRIVATE
    "-framework Metal"
    "-framework MetalPerformanceShaders"
)

