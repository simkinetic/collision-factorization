#include "mex.h"

// Signature: mex_call(@dense_tensor_kernel_mex, Q_out, f, C_dense, N_terms)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 1. Extract raw pointers
    double* Q = mxGetPr(prhs[0]);
    double* f = mxGetPr(prhs[1]);
    double* C = mxGetPr(prhs[2]);

    // 2. Extract total degrees of freedom (N_terms = N_Q * K_len)
    // N itself always fits in 32 bits, but N^2 and N^3 do not: the dense tensor
    // has N^3 entries, so at N = 1290 the largest linear index already reaches
    // 2^31. Performed in `int`, the offset (j + k*N)*N wraps negative and the
    // kernel dereferences a wild pointer -- a segmentation fault (not a
    // catchable MATLAB error) reproduced at N = 1521 (L_max=12, I_max=2).
    // All linear-index arithmetic is therefore done in size_t. Note the
    // multiplications must themselves be evaluated in the wide type: computing
    // (j + k*N)*N in `int` and promoting the result afterwards would still
    // overflow, so the loop counters are size_t rather than int.
    const size_t N = (size_t)mxGetScalar(prhs[3]);

    // 3. Dense 3D Contraction
    // Optimized for MATLAB's Column-Major memory layout: C[i + j*N + k*N^2]
    // The innermost loop MUST be 'i' to ensure perfectly contiguous memory reads.

    for (size_t k = 0; k < N; ++k) {
        double fk = f[k];
        for (size_t j = 0; j < N; ++j) {
            double fj_fk = f[j] * fk;

            // Pointer to the start of the 'i' column for this (j,k) pair
            const double* C_ptr = &C[(j + k * N) * N];

            // "Golden Loop" - Perfectly contiguous read of C and accumulation into Q
            for (size_t i = 0; i < N; ++i) {
                Q[i] += C_ptr[i] * fj_fk;
            }
        }
    }
}