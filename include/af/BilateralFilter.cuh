#ifndef BILATERALFILTER_CUH
#define BILATERALFILTER_CUH

namespace af {

void bilateralFilterOpmSafe(float* output, const float* input, int height, int width, int r, double sI, double sS);

void bilateralFilterTextureOpmShared(float* output, const float* input, int height, int width, int r, double sI, double sS, float thr);

}  // namespace af

#endif