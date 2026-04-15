#include "mex.h"
#include <vector>
#include <algorithm>

// Signature: q1_sliced_mex(Q_out, f, labels_sorted, vals_sorted, ic_map_sorted, R_tensor, N_Q, K_len)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double* Q      = mxGetPr(prhs[0]);
    double* f      = mxGetPr(prhs[1]);
    double* labels = mxGetPr(prhs[2]);
    double* vals   = mxGetPr(prhs[3]);
    double* ic_map = mxGetPr(prhs[4]);
    double* R      = mxGetPr(prhs[5]);

    int N_G = mxGetM(prhs[2]);
    int N_Q = (int)mxGetScalar(prhs[6]);
    int K   = (int)mxGetScalar(prhs[7]);
    
    int K2 = K * K;
    int K3 = K2 * K;

    // A tiny buffer for the K x K intermediate state (Fits entirely in L1 Cache/Registers)
    std::vector<double> Phi(K2, 0.0);

    int z = 0;
    while (z < N_G) {
        int t = (int)ic_map[z] - 1;
        
        // Find the block for this radial channel 't'
        int z_end_t = z;
        while (z_end_t < N_G && ((int)ic_map[z_end_t] - 1) == t) {
            z_end_t++;
        }

        // --- SLICE BY q1 ---
        int z_slice = z;
        while (z_slice < z_end_t) {
            int q1 = (int)labels[z_slice + 0 * N_G] - 1; // Note: Column 0 is q1
            
            // Find the end of this specific q1 slice
            int z_slice_end = z_slice;
            while (z_slice_end < z_end_t && ((int)labels[z_slice_end + 0 * N_G] - 1) == q1) {
                z_slice_end++;
            }

            // 1. Compute the K x K Phi matrix for this specific q1
            std::fill(Phi.begin(), Phi.end(), 0.0);
            for (int i = z_slice; i < z_slice_end; ++i) {
                int q2 = (int)labels[i + 1 * N_G] - 1;
                int q3 = (int)labels[i + 2 * N_G] - 1;
                double g = vals[i];

                for (int k3 = 0; k3 < K; ++k3) {
                    double f3 = f[q3 + k3 * N_Q];
                    for (int k2 = 0; k2 < K; ++k2) {
                        Phi[k2 + k3 * K] += g * f[q2 + k2 * N_Q] * f3;
                    }
                }
            }

            // 2. Immediately contract Phi with the Dense Radial Tensor R
            for (int k1 = 0; k1 < K; ++k1) {
                double q_val = 0.0;
                for (int k3 = 0; k3 < K; ++k3) {
                    for (int k2 = 0; k2 < K; ++k2) {
                        q_val += R[k1 + k2 * K + k3 * K2 + t * K3] * Phi[k2 + k3 * K];
                    }
                }
                Q[q1 + k1 * N_Q] += q_val;
            }

            // Move to the next q1 slice
            z_slice = z_slice_end;
        }
        z = z_end_t;
    }
}