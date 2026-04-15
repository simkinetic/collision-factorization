#include "mex.h"
#include <vector>

// Signature: radial_first_collision_kernel_mex(Q_out, f, labels_sorted, vals_sorted, ic_map_sorted, R_tensor, N_Q, K_len)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // 1. Extract raw pointers
    double* Q      = mxGetPr(prhs[0]);
    double* f      = mxGetPr(prhs[1]);
    double* labels = mxGetPr(prhs[2]);
    double* vals   = mxGetPr(prhs[3]);
    double* ic_map = mxGetPr(prhs[4]);
    double* R      = mxGetPr(prhs[5]);

    // 2. Extract dimensions
    int N_G = mxGetM(prhs[2]); // Total number of non-zero interactions
    int N_Q = (int)mxGetScalar(prhs[6]);
    int K   = (int)mxGetScalar(prhs[7]);
    
    int K2 = K * K;
    int K3 = K2 * K;

    // 3. Workspace buffer for Psi
    // Max possible size for a single radial channel is N_G * K. 
    // We allocate it once here to avoid dynamic allocation inside the loop.
    std::vector<double> Psi(N_G * K, 0.0);

    int z_start = 0;
    while (z_start < N_G) {
        int t = (int)ic_map[z_start] - 1;
        
        // Find the block of interactions sharing this radial channel 't'
        int z_end = z_start;
        while (z_end < N_G && ((int)ic_map[z_end] - 1) == t) {
            z_end++;
        }
        int N_zt = z_end - z_start;

        // =========================================================
        // STEP 1: Radial Contraction -> Psi[z, k1] = R * (f2 X f3)
        // =========================================================
        for (int z = 0; z < N_zt; ++z) {
            int global_z = z_start + z;
            int q2 = (int)labels[global_z + 1 * N_G] - 1;
            int q3 = (int)labels[global_z + 2 * N_G] - 1;

            for (int k1 = 0; k1 < K; ++k1) {
                double psi_val = 0.0;
                for (int k3 = 0; k3 < K; ++k3) {
                    double f3 = f[q3 + k3 * N_Q];
                    for (int k2 = 0; k2 < K; ++k2) {
                        double f2 = f[q2 + k2 * N_Q];
                        double r_val = R[k1 + k2 * K + k3 * K2 + t * K3];
                        
                        psi_val += r_val * f2 * f3;
                    }
                }
                Psi[z * K + k1] = psi_val;
            }
        }

        // =========================================================
        // STEP 2: Angular Routing -> Q[q1, k1] += Gamma * Psi
        // =========================================================
        for (int z = 0; z < N_zt; ++z) {
            int global_z = z_start + z;
            int q1 = (int)labels[global_z + 0 * N_G] - 1;
            double g = vals[global_z];

            for (int k1 = 0; k1 < K; ++k1) {
                Q[q1 + k1 * N_Q] += g * Psi[z * K + k1];
            }
        }

        // Move to the next radial physics channel
        z_start = z_end;
    }
}