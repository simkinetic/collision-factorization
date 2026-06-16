#include "mex.h"
#include <cmath>
#include <vector>
#include <algorithm>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ========================================================================
// FAST INLINE MATH HELPERS
// ========================================================================

// NEW: Fills the entire table of radial polynomials for a specific speed 'v'
inline void fill_radial_table(int N_K, int L_max, double v, const double* RadialNorm, std::vector<double>& table) {
    for (int l = 0; l <= L_max; ++l) {
        double v2 = v * v;
        double al = (double)l + 0.5;
        
        double vl = 1.0;
        for (int p = 0; p < l; ++p) vl *= v; // Avoid std::pow in hot loops

        if (N_K > 0) table[0 + N_K * l] = RadialNorm[0 + N_K * l] * vl;
        if (N_K > 1) table[1 + N_K * l] = RadialNorm[1 + N_K * l] * (1.0 + al - v2) * vl;

        if (N_K > 2) {
            double L0 = 1.0;
            double L1 = 1.0 + al - v2;
            for (int k = 2; k < N_K; ++k) {
                int i = k - 1;
                double Lk = ((2.0 * i + 1.0 + al - v2) * L1 - (i + al) * L0) / (i + 1.0);
                table[k + N_K * l] = RadialNorm[k + N_K * l] * Lk * vl;
                L0 = L1;
                L1 = Lk;
            }
        }
    }
}

// We still need eval_radial for vp_mag1 and vp_mag2 because they change with eps
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

inline double legendre_P(int l, int m, double x) {
    double pmm = 1.0;
    if (m > 0) {
        double somx2 = std::sqrt(std::max((1.0 - x) * (1.0 + x), 0.0));
        double fact = 1.0;
        for (int i = 1; i <= m; ++i) { pmm *= -fact * somx2; fact += 2.0; }
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

// ========================================================================
// DATA STRUCTURES
// ========================================================================
struct ThreadScratch {
    std::vector<double> R_local;
    std::vector<double> Y_tmp1;
    std::vector<double> Y_tmp2;
    std::vector<double> Phi_loss1;
    std::vector<double> Phi_loss2;
    std::vector<double> Phi_gain1;
    std::vector<double> Phi_gain2;
    
    // NEW: Hoisted Radial Tables
    std::vector<double> Rad_large;
    std::vector<double> Rad_small;

    ThreadScratch(int N_K, int N_L, int N_Q, int L_max) {
        R_local.assign(N_K * N_K * N_K * N_L, 0.0);
        Y_tmp1.assign(N_Q, 0.0);
        Y_tmp2.assign(N_Q, 0.0);
        Phi_loss1.assign(N_K * N_L, 0.0);
        Phi_loss2.assign(N_K * N_L, 0.0);
        Phi_gain1.assign(N_K * N_L, 0.0);
        Phi_gain2.assign(N_K * N_L, 0.0);
        
        Rad_large.assign(N_K * (L_max + 1), 0.0);
        Rad_small.assign(N_K * (L_max + 1), 0.0);
    }
};

// ========================================================================
// MEX MAIN FUNCTION
// ========================================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 26) {
        mexErrMsgIdAndTxt("R_tensor:InvalidInput", "Exactly 26 inputs required.");
    }
    
    int K_max = (int)mxGetScalar(prhs[0]); int N_K = K_max + 1;
    int N_L = (int)mxGetScalar(prhs[1]);
    int N_Q = (int)mxGetScalar(prhs[2]);
    double alpha = mxGetScalar(prhs[3]);
    int L_max = (int)std::round(std::sqrt(N_Q)) - 1;
    
    const double* x_nodes = mxGetPr(prhs[4]); const double* W_x = mxGetPr(prhs[5]); int N_x = mxGetNumberOfElements(prhs[4]);
    const double* u1_nodes = mxGetPr(prhs[6]); const double* W_u1 = mxGetPr(prhs[7]); int N_u1 = mxGetNumberOfElements(prhs[6]);
    const double* t1_nodes = mxGetPr(prhs[8]); const double* W_t1 = mxGetPr(prhs[9]); int N_t1 = mxGetNumberOfElements(prhs[8]);
    const double* y2_nodes = mxGetPr(prhs[10]); const double* W_y2 = mxGetPr(prhs[11]); int N_y2 = mxGetNumberOfElements(prhs[10]);
    const double* t2_nodes = mxGetPr(prhs[12]); const double* W_t2 = mxGetPr(prhs[13]); int N_t2 = mxGetNumberOfElements(prhs[12]);
    const double* mu_chi = mxGetPr(prhs[14]); const double* W_chi = mxGetPr(prhs[15]); int N_chi = mxGetNumberOfElements(prhs[14]);
    const double* eps_vec = mxGetPr(prhs[16]); const double* W_eps = mxGetPr(prhs[17]); int N_eps = mxGetNumberOfElements(prhs[16]);
    
    const double* RadialNorm = mxGetPr(prhs[18]); 
    const double* SH_Norm = mxGetPr(prhs[19]);    
    const double* L_triplets = mxGetPr(prhs[20]); 
    const double* qi_valid_mat = mxGetPr(prhs[21]); 
    
    const double* P_loss_p1 = mxGetPr(prhs[22]); 
    const double* P_gain_p1 = mxGetPr(prhs[23]); 
    const double* P_loss_p2 = mxGetPr(prhs[24]); 
    const double* P_gain_p2 = mxGetPr(prhs[25]); 

    mwSize dims[4] = {(mwSize)N_K, (mwSize)N_K, (mwSize)N_K, (mwSize)N_L};
    plhs[0] = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
    double* R_tensor = mxGetPr(plhs[0]);
    
    const double eps_safe = 1e-15;

    std::vector<double> c_eps(N_eps, 0.0);
    std::vector<double> s_eps(N_eps, 0.0);
    for (int e = 0; e < N_eps; ++e) {
        c_eps[e] = std::cos(eps_vec[e]);
        s_eps[e] = std::sin(eps_vec[e]);
    }

    #pragma omp parallel
    {
        // NEW: Pass L_max to initialize the radial table buffers
        ThreadScratch tb(N_K, N_L, N_Q, L_max);

        #pragma omp for
        for (int i = 0; i < N_x; ++i) {
            double x_i = x_nodes[i];
            
            // =================================================================
            // PATCH 1: u is Outer, t is Inner (y = u*t)
            // =================================================================
            for (int u_idx = 0; u_idx < N_u1; ++u_idx) {
                double u = u1_nodes[u_idx];
                double v_large = std::sqrt(x_i / 2.0) * (1.0 + u);
                double v_small = std::sqrt(x_i / 2.0) * (1.0 - u);
                
                double Jac_u = 0.5 * W_x[i] * W_u1[u_idx] * u * std::exp(-x_i * u * u) * (v_large * v_large * v_small * v_small);
                
                // --- NEW: Populate Radial Tables once per outer loop (Patch 1) ---
                fill_radial_table(N_K, L_max, v_large, RadialNorm, tb.Rad_large);
                fill_radial_table(N_K, L_max, v_small, RadialNorm, tb.Rad_small);

                std::fill(tb.Phi_loss1.begin(), tb.Phi_loss1.end(), 0.0); 
                std::fill(tb.Phi_loss2.begin(), tb.Phi_loss2.end(), 0.0);
                std::fill(tb.Phi_gain1.begin(), tb.Phi_gain1.end(), 0.0); 
                std::fill(tb.Phi_gain2.begin(), tb.Phi_gain2.end(), 0.0);

                for (int t_idx = 0; t_idx < N_t1; ++t_idx) {
                    double t = t1_nodes[t_idx]; 
                    double y = u * t;
                    double Jac_t = W_t1[t_idx] * (4.0 * u * t);
                    
                    double c_beta = 1.0 - 2.0 * y * y;
                    double s_beta = std::sqrt(std::max(1.0 - c_beta * c_beta, eps_safe));
                    
                    double u_mag = std::sqrt(2.0 * x_i) * u * std::sqrt(1.0 + (1.0 - u * u) * t * t);
                    double B_val = std::pow(u_mag * std::sqrt(2.0), alpha) / std::pow(x_i, alpha / 2.0);
                    double loss_weight = B_val * 2.0 * M_PI * 2.0 * Jac_t;

                    for (int t_chan = 0; t_chan < N_L; ++t_chan) {
                        int l_1 = (int)L_triplets[t_chan + N_L * 0];
                        double loss_int = loss_weight * P_loss_p1[t_idx + N_t1 * (u_idx + N_u1 * t_chan)];
                        for (int k1 = 0; k1 < N_K; ++k1) {
                            // --- NEW: Array lookup instead of eval_radial ---
                            tb.Phi_loss1[k1 + N_K * t_chan] += tb.Rad_large[k1 + N_K * l_1] * loss_int;
                            tb.Phi_loss2[k1 + N_K * t_chan] += tb.Rad_small[k1 + N_K * l_1] * loss_int;
                        }
                    }

                    double U_x1 = v_small * s_beta; double U_z1 = v_large + v_small * c_beta;
                    double u_x1 = -v_small * s_beta; double u_z1 = v_large - v_small * c_beta;
                    double u_mag1 = std::sqrt(std::max(u_x1*u_x1 + u_z1*u_z1, eps_safe));
                    double z_scat_x1 = u_x1 / u_mag1; double z_scat_z1 = u_z1 / u_mag1;
                    
                    double U_x2 = v_large * s_beta; double U_z2 = v_small + v_large * c_beta;
                    double u_x2 = -v_large * s_beta; double u_z2 = v_small - v_large * c_beta;
                    double u_mag2 = std::sqrt(std::max(u_x2*u_x2 + u_z2*u_z2, eps_safe));
                    double z_scat_x2 = u_x2 / u_mag2; double z_scat_z2 = u_z2 / u_mag2;

                    for (int n = 0; n < N_chi; ++n) {
                        double c_chi = mu_chi[n]; 
                        double s_chi = std::sqrt(std::max(1.0 - c_chi * c_chi, 0.0));
                        double gain_base = W_chi[n] * B_val * Jac_t;

                        for (int e_idx = 0; e_idx < N_eps; ++e_idx) {
                            double c_eps_val = c_eps[e_idx]; 
                            double s_eps_val = s_eps[e_idx];
                            
                            double u_px = s_chi * c_eps_val; 
                            double u_py = s_chi * s_eps_val; 
                            double u_pz = c_chi;
                            
                            double up_x1 = u_px * z_scat_z1 + u_pz * z_scat_x1; 
                            double up_z1 = -u_px * z_scat_x1 + u_pz * z_scat_z1;
                            double vp_x1 = 0.5 * (U_x1 + u_mag1 * up_x1); 
                            double vp_y1 = 0.5 * (u_mag1 * u_py); 
                            double vp_z1 = 0.5 * (U_z1 + u_mag1 * up_z1);
                            double vp_mag1 = std::sqrt(std::max(vp_x1*vp_x1 + vp_y1*vp_y1 + vp_z1*vp_z1, eps_safe));
                            eval_SH(std::acos(std::max(std::min(vp_z1 / vp_mag1, 1.0), -1.0)), std::atan2(vp_y1, vp_x1), N_Q, SH_Norm, tb.Y_tmp1.data());
                            
                            double up_x2 = u_px * z_scat_z2 + u_pz * z_scat_x2; 
                            double up_z2 = -u_px * z_scat_x2 + u_pz * z_scat_z2;
                            double vp_x2 = 0.5 * (U_x2 + u_mag2 * up_x2); 
                            double vp_y2 = 0.5 * (u_mag2 * u_py); 
                            double vp_z2 = 0.5 * (U_z2 + u_mag2 * up_z2);
                            double vp_mag2 = std::sqrt(std::max(vp_x2*vp_x2 + vp_y2*vp_y2 + vp_z2*vp_z2, eps_safe));
                            eval_SH(std::acos(std::max(std::min(vp_z2 / vp_mag2, 1.0), -1.0)), std::atan2(vp_y2, vp_x2), N_Q, SH_Norm, tb.Y_tmp2.data());

                            for (int t_chan = 0; t_chan < N_L; ++t_chan) {
                                int l_1 = (int)L_triplets[t_chan + N_L * 0];
                                double g_val1 = 0.0; double g_val2 = 0.0;
                                
                                for (int idx = 0; idx < N_Q; ++idx) {
                                    int q_i = (int)qi_valid_mat[idx + N_Q * t_chan];
                                    if (q_i == -1) break; 
                                    q_i -= 1; 
                                    
                                    double w_ang = P_gain_p1[t_idx + N_t1 * (u_idx + N_u1 * (q_i + N_Q * t_chan))];
                                    g_val1 += tb.Y_tmp1[q_i] * w_ang; 
                                    g_val2 += tb.Y_tmp2[q_i] * w_ang;
                                }
                                double wg1 = g_val1 * gain_base * W_eps[e_idx];
                                double wg2 = g_val2 * gain_base * W_eps[e_idx];
                                
                                // (vp_mag1 and vp_mag2 change per inner angle, so they still use eval_radial)
                                for (int k1 = 0; k1 < N_K; ++k1) {
                                    double n_norm = RadialNorm[k1 + N_K * l_1];
                                    tb.Phi_gain1[k1 + N_K * t_chan] += wg1 * eval_radial(k1+1, l_1, vp_mag1, n_norm);
                                    tb.Phi_gain2[k1 + N_K * t_chan] += wg2 * eval_radial(k1+1, l_1, vp_mag2, n_norm);
                                }
                            }
                        }
                    }
                } // End Inner t-loop

                // Patch 1: Factorized Contraction
                for (int t_chan = 0; t_chan < N_L; ++t_chan) {
                    int l_2 = (int)L_triplets[t_chan + N_L * 1]; 
                    int l_3 = (int)L_triplets[t_chan + N_L * 2];
                    for (int k1 = 0; k1 < N_K; ++k1) {
                        double net1 = Jac_u * (tb.Phi_gain1[k1 + N_K * t_chan] - tb.Phi_loss1[k1 + N_K * t_chan]);
                        double net2 = Jac_u * (tb.Phi_gain2[k1 + N_K * t_chan] - tb.Phi_loss2[k1 + N_K * t_chan]);
                        
                        for (int k2 = 0; k2 < N_K; ++k2) {
                            // --- NEW: Array lookup instead of eval_radial ---
                            double r2_l = tb.Rad_large[k2 + N_K * l_2];
                            double r2_s = tb.Rad_small[k2 + N_K * l_2];
                            for (int k3 = 0; k3 < N_K; ++k3) {
                                // --- NEW: Array lookup instead of eval_radial ---
                                double r3_l = tb.Rad_large[k3 + N_K * l_3];
                                double r3_s = tb.Rad_small[k3 + N_K * l_3];
                                
                                int out_idx = k1 + N_K * (k2 + N_K * (k3 + N_K * t_chan));
                                tb.R_local[out_idx] += net1 * r2_l * r3_s + net2 * r2_s * r3_l;
                            }
                        }
                    }
                }
            } // End Patch 1

            // =================================================================
            // PATCH 2: y is Outer, t is Inner (u = y*t)
            // =================================================================
            for (int y_idx = 0; y_idx < N_y2; ++y_idx) {
                double y = y2_nodes[y_idx];
                double Jac_y = 0.5 * W_x[i] * W_y2[y_idx] * y * (4.0 * y);
                
                double c_beta = 1.0 - 2.0 * y * y;
                double s_beta = std::sqrt(std::max(1.0 - c_beta * c_beta, eps_safe));

                for (int t_idx = 0; t_idx < N_t2; ++t_idx) {
                    double t = t2_nodes[t_idx]; 
                    double u = y * t;
                    
                    double v_large = std::sqrt(x_i / 2.0) * (1.0 + u);
                    double v_small = std::sqrt(x_i / 2.0) * (1.0 - u);
                    
                    // --- NEW: Populate Radial Tables once per inner loop (Patch 2) ---
                    fill_radial_table(N_K, L_max, v_large, RadialNorm, tb.Rad_large);
                    fill_radial_table(N_K, L_max, v_small, RadialNorm, tb.Rad_small);
                    
                    double Jac_t = W_t2[t_idx] * std::exp(-x_i * u * u) * (v_large * v_large * v_small * v_small);
                    double u_mag = std::sqrt(2.0 * x_i) * y * std::sqrt(t * t + 1.0 - y * y * t * t);
                    double B_val = std::pow(u_mag * std::sqrt(2.0), alpha) / std::pow(x_i, alpha / 2.0);
                    double loss_weight = B_val * 2.0 * M_PI * 2.0 * Jac_t;

                    std::fill(tb.Phi_loss1.begin(), tb.Phi_loss1.end(), 0.0); 
                    std::fill(tb.Phi_loss2.begin(), tb.Phi_loss2.end(), 0.0);
                    std::fill(tb.Phi_gain1.begin(), tb.Phi_gain1.end(), 0.0); 
                    std::fill(tb.Phi_gain2.begin(), tb.Phi_gain2.end(), 0.0);

                    for (int t_chan = 0; t_chan < N_L; ++t_chan) {
                        int l_1 = (int)L_triplets[t_chan + N_L * 0];
                        double loss_int = loss_weight * P_loss_p2[y_idx + N_y2 * t_chan];
                        for (int k1 = 0; k1 < N_K; ++k1) {
                            // --- NEW: Array lookup instead of eval_radial ---
                            tb.Phi_loss1[k1 + N_K * t_chan] += tb.Rad_large[k1 + N_K * l_1] * loss_int;
                            tb.Phi_loss2[k1 + N_K * t_chan] += tb.Rad_small[k1 + N_K * l_1] * loss_int;
                        }
                    }

                    double U_x1 = v_small * s_beta; double U_z1 = v_large + v_small * c_beta;
                    double u_x1 = -v_small * s_beta; double u_z1 = v_large - v_small * c_beta;
                    double u_mag1 = std::sqrt(std::max(u_x1*u_x1 + u_z1*u_z1, eps_safe));
                    double z_scat_x1 = u_x1 / u_mag1; double z_scat_z1 = u_z1 / u_mag1;
                    
                    double U_x2 = v_large * s_beta; double U_z2 = v_small + v_large * c_beta;
                    double u_x2 = -v_large * s_beta; double u_z2 = v_small - v_large * c_beta;
                    double u_mag2 = std::sqrt(std::max(u_x2*u_x2 + u_z2*u_z2, eps_safe));
                    double z_scat_x2 = u_x2 / u_mag2; double z_scat_z2 = u_z2 / u_mag2;

                    for (int n = 0; n < N_chi; ++n) {
                        double c_chi = mu_chi[n]; 
                        double s_chi = std::sqrt(std::max(1.0 - c_chi * c_chi, 0.0));
                        double gain_base = W_chi[n] * B_val * Jac_t;

                        for (int e_idx = 0; e_idx < N_eps; ++e_idx) {
                            double c_eps_val = c_eps[e_idx]; 
                            double s_eps_val = s_eps[e_idx];
                            
                            double u_px = s_chi * c_eps_val; 
                            double u_py = s_chi * s_eps_val; 
                            double u_pz = c_chi;
                            
                            double up_x1 = u_px * z_scat_z1 + u_pz * z_scat_x1; 
                            double up_z1 = -u_px * z_scat_x1 + u_pz * z_scat_z1;
                            double vp_x1 = 0.5 * (U_x1 + u_mag1 * up_x1); 
                            double vp_y1 = 0.5 * (u_mag1 * u_py); 
                            double vp_z1 = 0.5 * (U_z1 + u_mag1 * up_z1);
                            double vp_mag1 = std::sqrt(std::max(vp_x1*vp_x1 + vp_y1*vp_y1 + vp_z1*vp_z1, eps_safe));
                            eval_SH(std::acos(std::max(std::min(vp_z1 / vp_mag1, 1.0), -1.0)), std::atan2(vp_y1, vp_x1), N_Q, SH_Norm, tb.Y_tmp1.data());
                            
                            double up_x2 = u_px * z_scat_z2 + u_pz * z_scat_x2; 
                            double up_z2 = -u_px * z_scat_x2 + u_pz * z_scat_z2;
                            double vp_x2 = 0.5 * (U_x2 + u_mag2 * up_x2); 
                            double vp_y2 = 0.5 * (u_mag2 * u_py); 
                            double vp_z2 = 0.5 * (U_z2 + u_mag2 * up_z2);
                            double vp_mag2 = std::sqrt(std::max(vp_x2*vp_x2 + vp_y2*vp_y2 + vp_z2*vp_z2, eps_safe));
                            eval_SH(std::acos(std::max(std::min(vp_z2 / vp_mag2, 1.0), -1.0)), std::atan2(vp_y2, vp_x2), N_Q, SH_Norm, tb.Y_tmp2.data());

                            for (int t_chan = 0; t_chan < N_L; ++t_chan) {
                                int l_1 = (int)L_triplets[t_chan + N_L * 0];
                                double g_val1 = 0.0; double g_val2 = 0.0;
                                
                                for (int idx = 0; idx < N_Q; ++idx) {
                                    int q_i = (int)qi_valid_mat[idx + N_Q * t_chan];
                                    if (q_i == -1) break; 
                                    q_i -= 1; 
                                    
                                    double w_ang = P_gain_p2[y_idx + N_y2 * (q_i + N_Q * t_chan)];
                                    g_val1 += tb.Y_tmp1[q_i] * w_ang; 
                                    g_val2 += tb.Y_tmp2[q_i] * w_ang;
                                }
                                double wg1 = g_val1 * gain_base * W_eps[e_idx];
                                double wg2 = g_val2 * gain_base * W_eps[e_idx];
                                
                                for (int k1 = 0; k1 < N_K; ++k1) {
                                    double n_norm = RadialNorm[k1 + N_K * l_1];
                                    tb.Phi_gain1[k1 + N_K * t_chan] += wg1 * eval_radial(k1+1, l_1, vp_mag1, n_norm);
                                    tb.Phi_gain2[k1 + N_K * t_chan] += wg2 * eval_radial(k1+1, l_1, vp_mag2, n_norm);
                                }
                            }
                        }
                    }

                    // Patch 2: Factorized Contraction
                    for (int t_chan = 0; t_chan < N_L; ++t_chan) {
                        int l_2 = (int)L_triplets[t_chan + N_L * 1]; 
                        int l_3 = (int)L_triplets[t_chan + N_L * 2];
                        for (int k1 = 0; k1 < N_K; ++k1) {
                            double net1 = Jac_y * (tb.Phi_gain1[k1 + N_K * t_chan] - tb.Phi_loss1[k1 + N_K * t_chan]);
                            double net2 = Jac_y * (tb.Phi_gain2[k1 + N_K * t_chan] - tb.Phi_loss2[k1 + N_K * t_chan]);
                            
                            for (int k2 = 0; k2 < N_K; ++k2) {
                                // --- NEW: Array lookup instead of eval_radial ---
                                double r2_l = tb.Rad_large[k2 + N_K * l_2];
                                double r2_s = tb.Rad_small[k2 + N_K * l_2];
                                for (int k3 = 0; k3 < N_K; ++k3) {
                                    // --- NEW: Array lookup instead of eval_radial ---
                                    double r3_l = tb.Rad_large[k3 + N_K * l_3];
                                    double r3_s = tb.Rad_small[k3 + N_K * l_3];
                                    
                                    int out_idx = k1 + N_K * (k2 + N_K * (k3 + N_K * t_chan));
                                    tb.R_local[out_idx] += net1 * r2_l * r3_s + net2 * r2_s * r3_l;
                                }
                            }
                        }
                    }
                } // End Inner t-loop
            } // End Patch 2
        }

        // Thread-safe reduction into global R_tensor
        #pragma omp critical
        {
            for (int i = 0; i < N_K * N_K * N_K * N_L; ++i) {
                R_tensor[i] += tb.R_local[i];
            }
        }
    }
}