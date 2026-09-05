/**
 * nnpol_policy.c — the network interpreter: a dense MLP forward pass over
 * a slot's weights, dims and activation read from its header at run time.
 *
 * Kernel and layout are the exporter's const-fp32 ones (export_policy_c
 * _emit_forward output_major=True / _const_payload), transcribed from the
 * generated policy.c into loops over the header's layerDims: each kernel
 * stored TRANSPOSED (out, in) so an output's weights are contiguous, the
 * biases after it, and four outputs computed per pass so one activation
 * load feeds four multiply-accumulates while four sequential weight
 * streams keep the F405's flash prefetch working. Reading the weights
 * straight from the flash slot is what keeps the RAM cost at two 512-byte
 * activation buffers on the caller's stack.
 *
 * Pure C: host-testable, no firmware includes.
 */
#include <string.h>

#include "nnpol.h"

// crazyflie-firmware builds everything at -Os, which neither unrolls nor
// schedules the inner loops below; this translation unit is the one hot
// loop in the app and is worth the code size. The whole file is covered so
// fast_tanhf inlines into the layer loop (GCC refuses to inline across
// differing optimize attributes).
#pragma GCC optimize ("O2,unroll-loops")

/* The exporter's fast_tanhf, verbatim (export_policy_c._FAST_TANH_C): a
 * rational approximation whose error is below float32 resolution, so the
 * host numpy reference and the firmware agree to ~1e-7. */
static inline float fast_tanhf(float x)
{
  const float clamp = 7.90531110763549805f;
  if (x > clamp) { x = clamp; }
  if (x < -clamp) { x = -clamp; }
  if (x < 0.0004f && x > -0.0004f) { return x; }
  const float x2 = x * x;
  float p = -2.76076847742355e-16f;
  p = p * x2 + 2.00018790482477e-13f;
  p = p * x2 + -8.60467152213735e-11f;
  p = p * x2 + 5.12229709037114e-08f;
  p = p * x2 + 1.48572235717979e-05f;
  p = p * x2 + 6.37261928875436e-04f;
  p = p * x2 + 4.89352455891786e-03f;
  p = p * x;
  float q = 1.19825839466702e-06f;
  q = q * x2 + 1.18534705686654e-04f;
  q = q * x2 + 2.26843463243900e-03f;
  q = q * x2 + 4.89352518554385e-03f;
  return p / q;
}

static inline float reluf(float x) { return x > 0.0f ? x : 0.0f; }

/* One fp16 weight -> float. On the target GCC's __fp16 (the firmware
 * builds with -mfp16-format=ieee) is a single vcvtb.f32.f16 after the
 * halfword load; the host parity build has no such type and converts in
 * software. Both are exact - every half is representable in binary32 -
 * so the two paths agree bit for bit. */
#if defined(__ARM_FP16_FORMAT_IEEE)
static inline float h2f(uint16_t h)
{
  __fp16 v;
  memcpy(&v, &h, sizeof(v));
  return (float)v;
}
#else
static inline float h2f(uint16_t h)
{
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t bits;
  if (exp == 0u) {
    if (mant == 0u) {
      bits = sign;
    } else {
      int32_t e = 127 - 15 + 1;
      while ((mant & 0x400u) == 0u) { mant <<= 1; e--; }
      mant &= 0x3FFu;
      bits = sign | ((uint32_t)e << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &bits, sizeof(f));
  return f;
}
#endif

static inline float activate(uint16_t kind, float x)
{
  return kind == NNPOL_ACT_RELU ? reluf(x) : fast_tanhf(x);
}

#define BLOCK 4

void nnpolForward(const nnpolPolicy_t* p, const float* obs, float act[NNPOL_ACTION_DIM])
{
  const nnpolSlotHeader_t* h = p->hdr;
  const uint16_t numLayers = h->numLayers;
  const uint16_t activation = h->activation;
  float bufA[NNPOL_MAX_HIDDEN], bufB[NNPOL_MAX_HIDDEN];
  const float* in = obs;
  const uint16_t* w = p->weights;

  for (int l = 0; l < numLayers; l++) {
    const int inDim = h->layerDims[l];
    const int outDim = h->layerDims[l + 1];
    const int isLast = (l == numLayers - 1);
    const uint16_t* kernel = w;
    const uint16_t* bias = w + inDim * outDim;
    w = bias + outDim;
    float* out = isLast ? act : ((l & 1) ? bufB : bufA);

    int j = 0;
    for (; j + BLOCK <= outDim; j += BLOCK) {
      const uint16_t* w0 = kernel + (j + 0) * inDim;
      const uint16_t* w1 = kernel + (j + 1) * inDim;
      const uint16_t* w2 = kernel + (j + 2) * inDim;
      const uint16_t* w3 = kernel + (j + 3) * inDim;
      float a0 = h2f(bias[j + 0]), a1 = h2f(bias[j + 1]);
      float a2 = h2f(bias[j + 2]), a3 = h2f(bias[j + 3]);
      for (int k = 0; k < inDim; k++) {
        const float x = in[k];
        a0 += x * h2f(w0[k]); a1 += x * h2f(w1[k]);
        a2 += x * h2f(w2[k]); a3 += x * h2f(w3[k]);
      }
      if (isLast) {
        /* Final layer: raw linear output, no activation (models.Actor's
         * mean is never activation-squashed); the caller clips. */
        out[j + 0] = a0; out[j + 1] = a1; out[j + 2] = a2; out[j + 3] = a3;
      } else {
        out[j + 0] = activate(activation, a0);
        out[j + 1] = activate(activation, a1);
        out[j + 2] = activate(activation, a2);
        out[j + 3] = activate(activation, a3);
      }
    }
    for (; j < outDim; j++) {
      const uint16_t* wj = kernel + j * inDim;
      float acc = h2f(bias[j]);
      for (int k = 0; k < inDim; k++) {
        acc += in[k] * h2f(wj[k]);
      }
      out[j] = isLast ? acc : activate(activation, acc);
    }
    in = out;
  }
}
