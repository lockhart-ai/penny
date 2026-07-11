#include <metal_stdlib>
using namespace metal;

kernel void cosineDistance(
    device const float *query [[buffer(0)]],
    device const float *candidates [[buffer(1)]],
    device float *distances [[buffer(2)]],
    constant uint &dimension [[buffer(3)]],
    constant uint &candidateCount [[buffer(4)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= candidateCount) {
        return;
    }

    device const float *candidate = candidates + (index * dimension);
    float dot = 0.0;
    float queryNorm = 0.0;
    float candidateNorm = 0.0;
    for (uint component = 0; component < dimension; component++) {
        float queryValue = query[component];
        float candidateValue = candidate[component];
        dot += queryValue * candidateValue;
        queryNorm += queryValue * queryValue;
        candidateNorm += candidateValue * candidateValue;
    }

    if (queryNorm == 0.0 || candidateNorm == 0.0) {
        distances[index] = 2.0;
        return;
    }

    float similarity = dot / (sqrt(queryNorm) * sqrt(candidateNorm));
    distances[index] = 1.0 - similarity;
}
