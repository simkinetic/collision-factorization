#include "mex.h"

// Signature: mex_naive_eval(Q_out, f, gaunt_labels, gaunt_vals, ic_map, R_tensor, N_Q, K_len)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 1. Extract raw pointers (Pass-by-Reference modification on prhs[0])
    double* Q      = mxGetPr(prhs[0]);
    double* f      = mxGetPr(prhs[1]);
    double* labels = mxGetPr(prhs[2]);
    double* vals   = mxGetPr(prhs[3]);
    double* ic_map = mxGetPr(prhs[4]);
    double* R      = mxGetPr(prhs[5]);

    // 2. Extract dimensions
    int N_G   = mxGetM(prhs[2]); // Number of z-channels
    int N_Q   = (int)mxGetScalar(prhs[6]);
    int K_len = (int)mxGetScalar(prhs[7]);
    
    // Precompute tensor strides for 1D memory access
    int K2 = K_len * K_len;
    int K3 = K2 * K_len;

    // 3. Naive Loop: Execute physics per z-channel
    for (int z = 0; z < N_G; ++z) {
        // MATLAB is 1-indexed, C++ is 0-indexed
        int q1 = (int)labels[z + 0 * N_G] - 1;
        int q2 = (int)labels[z + 1 * N_G] - 1;
        int q3 = (int)labels[z + 2 * N_G] - 1;
        int t  = (int)ic_map[z] - 1;
        double g = vals[z];

        // 3D Radial Loop
        for (int k2 = 0; k2 < K_len; ++k2) {
            for (int k3 = 0; k3 < K_len; ++k3) {
                // Precalculate the trial outer-product
                double f2f3 = f[q2 + k2 * N_Q] * f[q3 + k3 * N_Q];
                
                for (int k1 = 0; k1 < K_len; ++k1) {
                    // R is size [K, K, K, N_T]
                    double r_val = R[k1 + k2 * K_len + k3 * K2 + t * K3];
                    
                    // Accumulate directly into Q pointer
                    Q[q1 + k1 * N_Q] += g * r_val * f2f3;
                }
            }
        }
    }
}