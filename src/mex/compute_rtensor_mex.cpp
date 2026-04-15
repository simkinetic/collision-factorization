#include "mex.h"
#include <cmath>
#include <vector>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

// ========================================================================
// FAST INLINE MATH HELPERS (Matches SpectralBasis.m perfectly)
// ========================================================================

// 1. Associated Laguerre Polynomial * v^l * Normalization
inline double eval_radial(int n_idx, int l, double v, double norm_factor) {
    int k = n_idx - 1; 
    double v2 = v * v;
    double al = (double)l + 0.5;
    
    double L_k = 0.0;
    if (k == 0) {
        L_k = 1.0; 
    } else if (k == 1) {
        L_k = 1.0 + al - v2;
    } else {
        double L0 = 1.0;
        double L1 = 1.0 + al - v2;
        for (int i = 1; i < k; ++i) {
            L_k = ((2.0 * i + 1.0 + al - v2) * L1 - (i + al) * L0) / (i + 1.0);
            L0 = L1;
            L1 = L_k;
        }
    }
    return norm_factor * L_k * std::pow(v, l); 
}

// Helper for Spherical Harmonics: Associated Legendre Polynomial P_l^m(x)
inline double legendre_P(int l, int m, double x) {
    double pmm = 1.0;
    if (m > 0) {
        double somx2 = std::sqrt(std::max((1.0 - x) * (1.0 + x), 0.0));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) {
            pmm *= -fact * somx2; 
            fact += 2.0;
        }
    }
    if (l == m) return pmm;
    
    double pmmp1 = x * (2 * m + 1) * pmm;
    if (l == m + 1) return pmmp1;
    
    double pll = 0.0;
    for (int ll = m + 2; ll <= l; ++ll) {
        pll = (x * (2 * ll - 1) * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    return pll;
}

// 2. Real Spherical Harmonics Y_l^m(theta, phi)
inline void eval_SH(double theta, double phi, int N_Q, const double* SH_Norm, double* Y_out) {
    int L_max = (int)std::round(std::sqrt(N_Q)) - 1;
    double x = std::cos(theta);
    
    for (int l = 0; l <= L_max; ++l) {
        for (int m = 0; m <= l; ++m) {
            double P = legendre_P(l, m, x);
            
            if (m == 0) {
                int q_idx = l * l + l; 
                Y_out[q_idx] = SH_Norm[q_idx] * P;
            } else {
                int q_pos = l * l + l + m;
                int q_neg = l * l + l - m;
                
                Y_out[q_pos] = SH_Norm[q_pos] * P * std::cos(m * phi);
                Y_out[q_neg] = SH_Norm[q_neg] * P * std::sin(m * phi);
            }
        }
    }
}

// 3. Native C++ Scattering Kernel (VHS Model)
// UPDATED: Now utilizes alpha and applies sqrt(2) scaling for physical velocity
inline double exact_kernel(double u_mag, double alpha) {
    return std::pow(u_mag * std::sqrt(2.0), alpha); 
}

// ========================================================================
// MEX MAIN FUNCTION
// ========================================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    // UPDATED: Check for 22 inputs (21 grids/weights + 1 alpha parameter)
    if (nrhs != 22) mexErrMsgIdAndTxt("R_tensor:InvalidInput", "Exactly 22 inputs required (including alpha).");
    
    // 1. EXTRACT SCALAR DIMENSIONS AND PARAMETERS
    int N_K     = (int)mxGetScalar(prhs[0]);
    int N_L     = (int)mxGetScalar(prhs[1]);
    int N_rad   = (int)mxGetScalar(prhs[2]);
    int N_theta = (int)mxGetScalar(prhs[3]);
    int N_chi   = (int)mxGetScalar(prhs[4]);
    int N_eps   = (int)mxGetScalar(prhs[5]);
    int N_Q     = (int)mxGetScalar(prhs[6]);
    
    // NEW: Extract alpha parameter from MATLAB
    double alpha_kernel = mxGetScalar(prhs[21]);
    
    int N_pts = N_theta * N_chi * N_eps;
    int L_max = (int)std::round(std::sqrt(N_Q)) - 1;

    // 2. EXTRACT RAW POINTERS FROM MATLAB INPUTS
    const double* v_nodes  = mxGetPr(prhs[7]);
    const double* w_nodes  = mxGetPr(prhs[8]);
    const double* W_vw     = mxGetPr(prhs[9]);
    const double* mu_theta = mxGetPr(prhs[10]);
    const double* mu_chi   = mxGetPr(prhs[11]);
    const double* eps_vec  = mxGetPr(prhs[12]);
    const double* W_Total  = mxGetPr(prhs[13]);
    const double* W_sphere = mxGetPr(prhs[14]);
    const double* W_theta_3D = mxGetPr(prhs[15]);
    
    const double* R_table        = mxGetPr(prhs[16]);
    const double* L_triplets_dbl = mxGetPr(prhs[17]);
    const double* P_loss_pre     = mxGetPr(prhs[18]);
    const double* P_gain_weights = mxGetPr(prhs[19]);
    const double* qi_valid_mat   = mxGetPr(prhs[20]);
    
    const double eps_safeguard = 1e-15;

    // 3. ALLOCATE THE OUTPUT TENSOR IN C++
    mwSize dims[4] = {(mwSize)N_K, (mwSize)N_K, (mwSize)N_K, (mwSize)N_L};
    plhs[0] = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
    double* R_tensor = mxGetPr(plhs[0]);

    // ====================================================================
    // 4. PRECOMPUTE NORMALIZATION CONSTANTS
    // ====================================================================
    std::vector<double> RadialNorm(N_K * (L_max + 1), 0.0);
    for (int n_idx = 1; n_idx <= N_K; ++n_idx) {
        int k = n_idx - 1;
        for (int l = 0; l <= L_max; ++l) {
            double al = (double)l + 0.5;
            double ln_M_ii = -std::log(2.0) + std::lgamma(k + al + 1.0) - std::lgamma(k + 1.0);
            RadialNorm[(n_idx - 1) * (L_max + 1) + l] = std::exp(-0.5 * ln_M_ii);
        }
    }
    std::vector<double> SH_Norm(N_Q, 0.0);
    for (int l = 0; l <= L_max; ++l) {
        for (int m = -l; m <= l; ++m) {
            int abs_m = std::abs(m);
            double base_norm = std::sqrt( ((2.0 * l + 1.0) / (4.0 * M_PI)) * std::exp(std::lgamma(l - abs_m + 1.0) - std::lgamma(l + abs_m + 1.0)) );
            int q_idx = l * l + l + m;
            if (m == 0) SH_Norm[q_idx] = base_norm;
            else        SH_Norm[q_idx] = M_SQRT2 * base_norm;
        }
    }

    // ====================================================================
    // 5. THE COMPUTATIONAL CORE
    // ====================================================================
    for (int a = 0; a < N_rad; ++a) {
        double v = v_nodes[a];
        
        std::vector<double> Y_temp(N_Q, 0.0);
        std::vector<double> R_gain_eval(N_pts * (L_max + 1) * N_K, 0.0);
        std::vector<double> Y_all_vp_4D(N_pts * N_Q, 0.0);
        std::vector<double> P_gain(N_pts, 0.0);
        std::vector<double> S_gain_all(N_K, 0.0);
        std::vector<double> I_loss_inner(N_theta, 0.0);
        std::vector<double> B_W_Total(N_pts, 0.0);
        
        for (int b = 0; b < N_rad; ++b) {
            double w = w_nodes[b];
            double weight_vw = W_vw[a + N_rad * b]; 
            
            std::fill(I_loss_inner.begin(), I_loss_inner.end(), 0.0);
            
            for (int e_idx = 0; e_idx < N_eps; ++e_idx) {
                double eps_val = eps_vec[e_idx];
                double cos_e = std::cos(eps_val);
                double sin_e = std::sin(eps_val);
                
                for (int c_idx = 0; c_idx < N_chi; ++c_idx) {
                    double mu_c = mu_chi[c_idx];
                    double sin_c = std::sqrt(std::max(1.0 - mu_c * mu_c, 0.0));
                    
                    double u_prime_x_scat = sin_c * cos_e;
                    double u_prime_y_scat = sin_c * sin_e;
                    double u_prime_z_scat = mu_c;
                    
                    for (int th_idx = 0; th_idx < N_theta; ++th_idx) {
                        int pt = th_idx + N_theta * (c_idx + N_chi * e_idx);
                        
                        double mu_t = mu_theta[th_idx];
                        double sin_t = std::sqrt(std::max(1.0 - mu_t * mu_t, 0.0));
                        
                        double U_x = w * sin_t;
                        double U_z = v + w * mu_t;
                        double u_x = -w * sin_t;
                        double u_z = v - w * mu_t;
                        
                        double u_mag = std::sqrt(std::max(u_x * u_x + u_z * u_z, eps_safeguard));
                        
                        double z_scat_x = u_x / u_mag;
                        double z_scat_z = u_z / u_mag;
                        double x_scat_x = z_scat_z;
                        double x_scat_z = -z_scat_x;
                        
                        double u_prime_x = u_prime_x_scat * x_scat_x + u_prime_z_scat * z_scat_x;
                        double u_prime_y = u_prime_y_scat; 
                        double u_prime_z = u_prime_x_scat * x_scat_z + u_prime_z_scat * z_scat_z;
                        
                        double v_prime_x = 0.5 * (U_x + u_mag * u_prime_x);
                        double v_prime_y = 0.5 * (      u_mag * u_prime_y);
                        double v_prime_z = 0.5 * (U_z + u_mag * u_prime_z);
                        
                        double v_prime_mag = std::sqrt(std::max(v_prime_x * v_prime_x + v_prime_y * v_prime_y + v_prime_z * v_prime_z, eps_safeguard));
                        
                        double ratio = std::max(std::min(v_prime_z / v_prime_mag, 1.0), -1.0);
                        double theta_vp = std::acos(ratio);
                        double phi_vp   = std::atan2(v_prime_y, v_prime_x);
                        
                        // UPDATED: Now passing alpha_kernel
                        double B_val = exact_kernel(u_mag, alpha_kernel);
                        
                        I_loss_inner[th_idx] += B_val * W_sphere[c_idx + N_chi * e_idx];
                        B_W_Total[pt] = B_val * W_Total[pt];
                        
                        eval_SH(theta_vp, phi_vp, N_Q, SH_Norm.data(), Y_temp.data());
                        for (int q = 0; q < N_Q; ++q) {
                            Y_all_vp_4D[pt + N_pts * q] = Y_temp[q]; 
                        }
                        
                        for (int l_idx = 0; l_idx <= L_max; ++l_idx) {
                            for (int n_idx = 1; n_idx <= N_K; ++n_idx) {
                                int n = n_idx - 1;
                                double norm_val = RadialNorm[n * (L_max + 1) + l_idx];
                                int base_idx = N_pts * (l_idx + (L_max + 1) * n);
                                R_gain_eval[pt + base_idx] = eval_radial(n_idx, l_idx, v_prime_mag, norm_val); 
                            }
                        }
                    }
                }
            }
            
            for (int t = 0; t < N_L; ++t) {
                int l_i = (int)L_triplets_dbl[t + N_L * 0];
                int l_j = (int)L_triplets_dbl[t + N_L * 1];
                int l_k = (int)L_triplets_dbl[t + N_L * 2];
                
                std::fill(P_gain.begin(), P_gain.end(), 0.0);
                
                for (int idx = 0; idx < N_Q; ++idx) {
                    int q_i_matlab = (int)qi_valid_mat[idx + N_Q * t];
                    if (q_i_matlab == -1) break; 
                    int q_i = q_i_matlab - 1; 
                    
                    for (int pt = 0; pt < N_pts; ++pt) {
                        int theta_idx = pt % N_theta; 
                        double weight = P_gain_weights[theta_idx + N_theta * (q_i + N_Q * t)];
                        P_gain[pt] += Y_all_vp_4D[pt + N_pts * q_i] * weight;
                    }
                }
                
                double S_loss_angular = 0.0;
                for (int th_idx = 0; th_idx < N_theta; ++th_idx) {
                    S_loss_angular += I_loss_inner[th_idx] * P_loss_pre[th_idx + N_theta * t] * W_theta_3D[th_idx];
                }
                
                for (int n = 0; n < N_K; ++n) {
                    double s_gain = 0.0;
                    int base_idx = N_pts * (l_i + (L_max + 1) * n);
                    for (int pt = 0; pt < N_pts; ++pt) {
                        s_gain += R_gain_eval[pt + base_idx] * P_gain[pt] * B_W_Total[pt];
                    }
                    S_gain_all[n] = s_gain;
                }
                
                for (int n_k = 0; n_k < N_K; ++n_k) {
                    double r_k_val = R_table[n_k + N_K * (l_k + (L_max + 1) * b)];
                    for (int n_j = 0; n_j < N_K; ++n_j) {
                        double r_j_val = R_table[n_j + N_K * (l_j + (L_max + 1) * a)];
                        double scalar_jk = r_j_val * r_k_val;
                        for (int n_i = 0; n_i < N_K; ++n_i) {
                            double r_loss_val = R_table[n_i + N_K * (l_i + (L_max + 1) * a)];
                            double net_s = (S_gain_all[n_i] - S_loss_angular * r_loss_val) * weight_vw;
                            
                            int out_idx = n_i + N_K * (n_j + N_K * (n_k + N_K * t));
                            R_tensor[out_idx] += net_s * scalar_jk;
                        }
                    }
                }
            }
        }
    }
}