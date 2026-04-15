#include "mex.h"

// Signature: mex_call(@dense_tensor_kernel_mex, Q_out, f, C_dense, N_terms)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 1. Extract raw pointers
    double* Q = mxGetPr(prhs[0]);
    double* f = mxGetPr(prhs[1]);
    double* C = mxGetPr(prhs[2]);

    // 2. Extract total degrees of freedom (N_terms = N_Q * K_len)
    int N = (int)mxGetScalar(prhs[3]);

    // 3. Dense 3D Contraction
    // Optimized for MATLAB's Column-Major memory layout: C[i + j*N + k*N^2]
    // The innermost loop MUST be 'i' to ensure perfectly contiguous memory reads.
    
    for (int k = 0; k < N; ++k) {
        double fk = f[k];
        for (int j = 0; j < N; ++j) {
            double fj_fk = f[j] * fk;
            
            // Pointer to the start of the 'i' column for this (j,k) pair
            const double* C_ptr = &C[(j + k * N) * N];
            
            // "Golden Loop" - Perfectly contiguous read of C and accumulation into Q
            for (int i = 0; i < N; ++i) {
                Q[i] += C_ptr[i] * fj_fk;
            }
        }
    }
}