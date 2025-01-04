#include <iostream>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

int main() {
    @autoreleasepool {
        // Create Metal device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nullptr) {
            std::cerr << "Metal is not supported on this device.\n";
            return -1;
        }

        // Define vector size
        const size_t vectorSize = 4;

        // Input vectors
        float vectorA[vectorSize] = {1.0, 2.0, 3.0, 4.0};
        float vectorB[vectorSize] = {5.0, 6.0, 7.0, 8.0};
        float resultVector[vectorSize] = {0.0, 0.0, 0.0, 0.0};

        // Create Metal buffers
        id<MTLBuffer> bufferA = [device newBufferWithBytes:vectorA
                                                    length:vectorSize * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:vectorB
                                                    length:vectorSize * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferResult = [device newBufferWithBytes:resultVector
                                                         length:vectorSize * sizeof(float)
                                                        options:MTLResourceStorageModeShared];

        // Create matrix and vector descriptors
        MPSMatrixDescriptor *matrixDescriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:vectorSize
                             columns:1
                             rowBytes:vectorSize * sizeof(float)
                              dataType:MPSDataTypeFloat32];

        MPSVectorDescriptor *vectorDescriptor = [MPSVectorDescriptor
            vectorDescriptorWithLength:vectorSize
                              dataType:MPSDataTypeFloat32];

        // Wrap buffers in MPS objects
        MPSMatrix *matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:matrixDescriptor];
        MPSVector *vectorBObj = [[MPSVector alloc] initWithBuffer:bufferB descriptor:vectorDescriptor];
        MPSVector *vectorResultObj = [[MPSVector alloc] initWithBuffer:bufferResult descriptor:vectorDescriptor];

        // Create MPSMatrixVectorMultiplication
        MPSMatrixVectorMultiplication *matrixVectorMultiplication = [[MPSMatrixVectorMultiplication alloc]
            initWithDevice:device
                  transpose:NO
                       rows:vectorSize
                    columns:1
                      alpha:1.0
                       beta:0.0];

        // Command queue and command buffer
        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

        // Encode MPS operation
        [matrixVectorMultiplication encodeToCommandBuffer:commandBuffer
                                              inputMatrix:matrixA
                                              inputVector:vectorBObj
                                              resultVector:vectorResultObj];

        // Commit and wait for completion
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy results
        memcpy(resultVector, [bufferResult contents], vectorSize * sizeof(float));

        // Print results
        std::cout << "Result Vector: ";
        for (float i : resultVector) {
            std::cout << i << " ";
        }
        std::cout << '\n';
    }
    return 0;
}
