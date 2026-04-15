#include "mex.h"
#include <vector>
#include <algorithm>

// Signature: mex_kronecker_eval(Q_out, f, labels_sorted, vals_sorted, ic_map_sorted, R_tensor, N_Q, K_len)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 1. Extract raw pointers
    double* Q      = mxGetPr(prhs[0]);
    double* f      = mxGetPr(prhs[1]);
    double* labels = mxGetPr(prhs[2]);
    double* vals   = mxGetPr(prhs[3]);
    double* ic_map = mxGetPr(prhs[4]);
    double* R      = mxGetPr(prhs[5]);

    // 2. Extract dimensions
    int N_G = mxGetM(prhs[2]);
    int N_Q = (int)mxGetScalar(prhs[6]);
    int K   = (int)mxGetScalar(prhs[7]);
    
    int K2 = K * K;
    int K3 = K2 * K;

    // 3. Allocate intermediate Kronecker state (Phi)
    // Size: [N_Q, K, K], initialized to zero.
    std::vector<double> Phi(N_Q * K2, 0.0);

    int z = 0;
    while (z < N_G) {
        // Read the radial physics channel for this block
        int t = (int)ic_map[z] - 1;

        // --- STEP A: Sparse Angular Slicing ---
        // Accumulate trial functions for all z-channels sharing this t
        while (z < N_G && ((int)ic_map[z] - 1) == t) {
            int q1 = (int)labels[z + 0 * N_G] - 1;
            int q2 = (int)labels[z + 1 * N_G] - 1;
            int q3 = (int)labels[z + 2 * N_G] - 1;
            double g = vals[z];

            for (int k3 = 0; k3 < K; ++k3) {
                for (int k2 = 0; k2 < K; ++k2) {
                    // Phi[q1, k2, k3]
                    Phi[q1 + (k2 + k3 * K) * N_Q] += g * f[q2 + k2 * N_Q] * f[q3 + k3 * N_Q];
                }
            }
            z++; // Move to next channel
        }

        // --- STEP B: Dense Radial Contraction ---
        // Apply the R tensor to the accumulated Phi states
        for (int k3 = 0; k3 < K; ++k3) {
            for (int k2 = 0; k2 < K; ++k2) {
                for (int k1 = 0; k1 < K; ++k1) {
                    double r_val = R[k1 + k2 * K + k3 * K2 + t * K3];
                    
                    // Notice that 'q' is the innermost loop! 
                    // This creates perfectly contiguous, SIMD-vectorized memory access.
                    for (int q = 0; q < N_Q; ++q) {
                        Q[q + k1 * N_Q] += Phi[q + (k2 + k3 * K) * N_Q] * r_val;
                    }
                }
            }
        }

        // --- STEP C: Reset ---
        // Clear the Phi buffer for the next t-channel
        std::fill(Phi.begin(), Phi.end(), 0.0);
    }
}