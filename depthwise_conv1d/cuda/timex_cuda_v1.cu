#include <stdio.h>

// require T <= Tmax

template <typename F>
__global__ void kernel_forward(const F *__restrict__ const w, const F *__restrict__ const k, F *__restrict__ const x,
                               const F eps, const int B, const int C, const int T)
{
    const int i = blockIdx.y;
    const int t = threadIdx.x;

    __shared__ F ww[Tmax];
    __shared__ F kk[Tmax];
    ww[t] = w[(i % C) * T + t];
    kk[t] = k[i * T + t];

    __syncthreads();

    F s = eps;
    const F *__restrict__ const www = ww + (T - 1) - t;
    for (int u = 0; u <= t; u++)
    {
        s += www[u] * kk[u];
    }
    x[i * T + t] = s;
}

template <typename F>
__global__ void kernel_backward(const F *__restrict__ const w, const F *__restrict__ const k, const F *__restrict__ const gwk,
                                F *__restrict__ const gw, F *__restrict__ const gk,
                                const int B, const int C, const int T)
{
    const int i = blockIdx.y;
    const int t = threadIdx.x;

    __shared__ F gg[Tmax];
    __shared__ F kk[Tmax];
    __shared__ F ww[Tmax];
    gg[t] = gwk[i * T + t];
    kk[t] = k[i * T + t];
    ww[t] = w[(i % C) * T + t];

    __syncthreads();

    F s = 0;
    const F *__restrict__ const ggk = gg + (T - 1) - t;
    for (int u = 0; u <= t; u++)
    {
        s += ggk[u] * kk[u];
    }
    gw[i * T + t] = s;

    s = 0;
    const F *__restrict__ const ggw = gg + (T - 1) + t;
    for (int u = t; u < T; u++)
    {
        s += ggw[-u] * ww[u];
    }
    gk[i * T + t] = s;
//  看前向传播输出了什么，以及哪些w参与了这个k[t]的计算，不清楚可以写一个简单脚本
     // import numpy as np
// h = np.arange(0,10)
// w = np.arange(10,20)
// hidx = 4
// s = 0
// for idx in range(10):
//     for i in range(idx):
//         s = s + h[i]*w[10 - idx + i]
//         if i == 2:
//             print(f"h_{i} and w_{10 - idx + i}, for {idx}")
// 　输出如下：
// h_2 and w_9, for 3
// h_2 and w_8, for 4
// h_2 and w_7, for 5
// h_2 and w_6, for 6
// h_2 and w_5, for 7
// h_2 and w_4, for 8
// h_2 and w_3, for 9
}

void cuda_forward(const float *w, const float *k, float *x, float eps, int B, int C, int T)
{
    dim3 gridDim(1, B * C);
    dim3 blockDim(T);
    kernel_forward<<<gridDim, blockDim>>>(w, k, x, eps, B, C, T);
}
void cuda_backward(const float *w, const float *k, const float *gwk, float *gw, float *gk, int B, int C, int T)
{
    dim3 gridDim(1, B * C);
    dim3 blockDim(T);
    kernel_backward<<<gridDim, blockDim>>>(w, k, gwk, gw, gk, B, C, T);
}
