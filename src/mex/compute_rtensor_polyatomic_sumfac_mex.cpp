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

inline void fill_radial_table(int N_K_rad, int L_max, double v, const double* RadialNorm, std::vector<double>& table) {
    for (int l = 0; l <= L_max; ++l) {
        double v2 = v * v;
        double al = (double)l + 0.5;
        double vl = 1.0;
        for (int p = 0; p < l; ++p) vl *= v; 

        if (N_K_rad > 0) table[0 + N_K_rad * l] = RadialNorm[0 + N_K_rad * l] * vl;
        if (N_K_rad > 1) table[1 + N_K_rad * l] = RadialNorm[1 + N_K_rad * l] * (1.0 + al - v2) * vl;

        if (N_K_rad > 2) {
            double L0 = 1.0; double L1 = 1.0 + al - v2;
            for (int k = 2; k < N_K_rad; ++k) {
                int i = k - 1;
                double Lk = ((2.0 * i + 1.0 + al - v2) * L1 - (i + al) * L0) / (i + 1.0);
                table[k + N_K_rad * l] = RadialNorm[k + N_K_rad * l] * Lk * vl;
                L0 = L1; L1 = Lk;
            }
        }
    }
}

inline void fill_internal_array(int N_I, double nu, double I_val, const double* InternalNorm, std::vector<double>& table) {
    if (N_I > 0) table[0] = InternalNorm[0]; 
    if (N_I > 1) table[1] = InternalNorm[1] * (1.0 + nu - I_val);
    
    if (N_I > 2) {
        double L0 = 1.0; double L1 = 1.0 + nu - I_val;
        for (int i = 2; i < N_I; ++i) {
            int k = i - 1;
            double Lk = ((2.0 * k + 1.0 + nu - I_val) * L1 - (k + nu) * L0) / (k + 1.0);
            table[i] = InternalNorm[i] * Lk;
            L0 = L1; L1 = Lk;
        }
    }
}

inline void eval_SH_fast(double x, double cphi, double sphi, int N_Q, const double* SH_Norm, double* Y_out) {
    int L_max = (int)std::round(std::sqrt(N_Q)) - 1;
    double somx2 = std::sqrt(std::max((1.0 - x) * (1.0 + x), 0.0));

    double cosm[256]; double sinm[256];
    cosm[0] = 1.0; sinm[0] = 0.0;
    if (L_max >= 1) { cosm[1] = cphi; sinm[1] = sphi; }
    
    for (int m = 2; m <= L_max; ++m) {
        cosm[m] = 2.0 * cphi * cosm[m - 1] - cosm[m - 2];
        sinm[m] = 2.0 * cphi * sinm[m - 1] - sinm[m - 2];
    }

    double pmm = 1.0;
    for (int m = 0; m <= L_max; ++m) {
        if (m > 0) pmm *= -(2.0 * m - 1.0) * somx2;
        double Plm2, Plm1, Pl; int base_mm = m * m + m;
        
        if (m == 0) { Y_out[base_mm] = SH_Norm[base_mm] * pmm; } 
        else {
            double nb = SH_Norm[base_mm + m] * pmm;
            Y_out[base_mm + m] = nb * cosm[m];
            Y_out[base_mm - m] = SH_Norm[base_mm - m] * pmm * sinm[m];
        }
        
        Plm1 = pmm;
        if (m + 1 <= L_max) {
            Pl = x * (2.0 * m + 1.0) * pmm;
            int base_l = (m + 1) * (m + 1) + (m + 1);
            
            double nb = SH_Norm[base_l + m] * Pl;
            Y_out[base_l + m] = (m == 0) ? nb : nb * cosm[m];
            if (m > 0) Y_out[base_l - m] = SH_Norm[base_l - m] * Pl * sinm[m];
            
            Plm2 = Plm1; Plm1 = Pl;
            for (int l = m + 2; l <= L_max; ++l) {
                Pl = (x * (2.0 * l - 1.0) * Plm1 - (l + m - 1.0) * Plm2) / (l - m);
                int bl = l * l + l;
                double nb_l = SH_Norm[bl + m] * Pl;
                Y_out[bl + m] = (m == 0) ? nb_l : nb_l * cosm[m];
                if (m > 0) Y_out[bl - m] = SH_Norm[bl - m] * Pl * sinm[m];
                Plm2 = Plm1; Plm1 = Pl;
            }
        }
    }
}

// ========================================================================
// DATA STRUCTURES
// ========================================================================

struct Config {
    int K_max, I_max, N_K, N_I, K_test, I_test, N_K_test, N_I_test, N_K_rad, N_I_rad, N_L, N_Q, L_max;
    double alpha, nu;
    
    const double *x_nodes, *W_x; int N_x;
    const double *u1_nodes, *W_u1; int N_u1;
    const double *t1_nodes, *W_t1; int N_t1;
    const double *y2_nodes, *W_y2; int N_y2;
    const double *t2_nodes, *W_t2; int N_t2;
    const double *mu_chi, *W_chi; int N_chi;
    const double *eps_vec, *W_eps; int N_eps;
    
    const double *I_nodes, *W_I; int N_I_nodes;
    const double *J_nodes, *W_J; int N_J_nodes;
    const double *r_nodes, *W_r; int N_r_nodes;
    const double *R_nodes, *W_R; int N_R_nodes;
    
    const double *RadialNorm, *InternalNorm, *SH_Norm, *L_triplets, *qi_valid_mat;
    const double *P_loss_p1, *P_gain_p1, *P_loss_p2, *P_gain_p2;
    
    const std::vector<double>* c_eps;
    const std::vector<double>* s_eps;
    double sum_WR, sum_Wr;
};

struct ThreadScratch {
    std::vector<double> R_local;
    std::vector<double> Y_tmp1, Y_tmp2, Rad_large, Rad_small, rad_vpmag1, rad_vpmag2;
    std::vector<double> H_I_test, H_J_test, H_I_trial, H_J_trial, H_Ip, H_Jp;
    std::vector<double> Phi_loss1, Phi_loss2, N_gain1, N_gain2, S_gain1, S_gain2, M_gain_I, M_gain_J, net1, net2;

    ThreadScratch(const Config& cfg) {
        R_local.assign(cfg.N_K_test * cfg.N_K * cfg.N_K * cfg.N_I_test * cfg.N_I * cfg.N_I * cfg.N_L, 0.0);
        Y_tmp1.assign(cfg.N_Q, 0.0); Y_tmp2.assign(cfg.N_Q, 0.0);
        
        Rad_large.assign(cfg.N_K_rad * (cfg.L_max + 1), 0.0); 
        Rad_small.assign(cfg.N_K_rad * (cfg.L_max + 1), 0.0);
        rad_vpmag1.assign(cfg.N_K_rad * (cfg.L_max + 1), 0.0); 
        rad_vpmag2.assign(cfg.N_K_rad * (cfg.L_max + 1), 0.0);
        
        H_I_test.assign(cfg.N_I_rad, 0.0); H_J_test.assign(cfg.N_I_rad, 0.0);
        H_I_trial.assign(cfg.N_I_rad, 0.0); H_J_trial.assign(cfg.N_I_rad, 0.0);
        H_Ip.assign(cfg.N_I_rad, 0.0); H_Jp.assign(cfg.N_I_rad, 0.0);
        
        M_gain_I.assign(cfg.N_I_test, 0.0); M_gain_J.assign(cfg.N_I_test, 0.0);
        
        Phi_loss1.assign(cfg.N_K_test * cfg.N_I_test * cfg.N_L, 0.0); 
        Phi_loss2.assign(cfg.N_K_test * cfg.N_I_test * cfg.N_L, 0.0);
        N_gain1.assign(cfg.N_K_test * cfg.N_I_test * cfg.N_L, 0.0); 
        N_gain2.assign(cfg.N_K_test * cfg.N_I_test * cfg.N_L, 0.0);
        S_gain1.assign(cfg.N_K_test * cfg.N_L, 0.0); 
        S_gain2.assign(cfg.N_K_test * cfg.N_L, 0.0);
        
        net1.assign(cfg.N_K_test, 0.0); net2.assign(cfg.N_K_test, 0.0);
    }
};

// ========================================================================
// POLYATOMIC ABSTRACTION (FUBINI CASCADE)
// ========================================================================

inline void compute_polyatomic_inner(const Config& cfg, ThreadScratch& tb, 
                                     double Jac_spatial, double u_mag, double x_i,
                                     double U_x1, double U_z1, double u_x1, double u_z1,
                                     double U_x2, double U_z2, double u_x2, double u_z2,
                                     const double* P_loss, const double* P_gain, int spatial_stride) {
    
    const double eps_safe = 1e-15;
    double u_mag1 = std::sqrt(std::max(u_x1*u_x1 + u_z1*u_z1, eps_safe));
    double z_scat_x1 = u_x1 / u_mag1; double z_scat_z1 = u_z1 / u_mag1;
    
    double u_mag2 = std::sqrt(std::max(u_x2*u_x2 + u_z2*u_z2, eps_safe));
    double z_scat_x2 = u_x2 / u_mag2; double z_scat_z2 = u_z2 / u_mag2;

    for (int I_idx = 0; I_idx < cfg.N_I_nodes; ++I_idx) {
        double I_val = cfg.I_nodes[I_idx];
        fill_internal_array(cfg.N_I_test, cfg.nu, I_val, cfg.InternalNorm, tb.H_I_test);
        fill_internal_array(cfg.N_I, cfg.nu, I_val, cfg.InternalNorm, tb.H_I_trial);

        for (int J_idx = 0; J_idx < cfg.N_J_nodes; ++J_idx) {
            double J_val = cfg.J_nodes[J_idx];
            fill_internal_array(cfg.N_I_test, cfg.nu, J_val, cfg.InternalNorm, tb.H_J_test); 
            fill_internal_array(cfg.N_I, cfg.nu, J_val, cfg.InternalNorm, tb.H_J_trial);

            double E = 0.5 * u_mag * u_mag + I_val + J_val;
            
            // Singularity scaling (x_i) ensures analytic transition probability across all interaction potentials
            double B_val = (std::pow(u_mag * std::sqrt(2.0), cfg.alpha) + std::pow(I_val + J_val, cfg.alpha / 2.0)) / std::pow(x_i, cfg.alpha / 2.0);
            
            // Exact Probability Mass Matching: Gain and Loss use identical un-normalized sum representations
            double factor = cfg.W_I[I_idx] * cfg.W_J[J_idx] * B_val * Jac_spatial;
            double loss_weight = factor * 4.0 * M_PI * cfg.sum_WR * cfg.sum_Wr;

            std::fill(tb.Phi_loss1.begin(), tb.Phi_loss1.end(), 0.0); 
            std::fill(tb.Phi_loss2.begin(), tb.Phi_loss2.end(), 0.0);
            
            for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                int l_1 = (int)cfg.L_triplets[t_chan + cfg.N_L * 0];
                double loss_int = loss_weight * P_loss[t_chan * spatial_stride];

                for (int i1 = 0; i1 < cfg.N_I_test; ++i1) {
                    for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                        int ph_idx = k1 + cfg.N_K_test * (i1 + cfg.N_I_test * t_chan);
                        tb.Phi_loss1[ph_idx] = loss_int * tb.Rad_large[k1 + cfg.N_K_rad * l_1] * tb.H_I_test[i1];
                        tb.Phi_loss2[ph_idx] = loss_int * tb.Rad_small[k1 + cfg.N_K_rad * l_1] * tb.H_J_test[i1];
                    }
                }
            }
            
            std::fill(tb.N_gain1.begin(), tb.N_gain1.end(), 0.0);
            std::fill(tb.N_gain2.begin(), tb.N_gain2.end(), 0.0);

            for (int R_idx = 0; R_idx < cfg.N_R_nodes; ++R_idx) {
                double R_val = cfg.R_nodes[R_idx];
                double u_prime = std::sqrt(2.0 * R_val * E);
                
                // Branch A: Integrate r
                std::fill(tb.M_gain_I.begin(), tb.M_gain_I.end(), 0.0);
                std::fill(tb.M_gain_J.begin(), tb.M_gain_J.end(), 0.0);
                for (int r_idx = 0; r_idx < cfg.N_r_nodes; ++r_idx) {
                    double r_val = cfg.r_nodes[r_idx];
                    double I_prime = r_val * (1.0 - R_val) * E;
                    double J_prime = (1.0 - r_val) * (1.0 - R_val) * E;

                    fill_internal_array(cfg.N_I_test, cfg.nu, I_prime, cfg.InternalNorm, tb.H_Ip);
                    fill_internal_array(cfg.N_I_test, cfg.nu, J_prime, cfg.InternalNorm, tb.H_Jp);

                    for (int i1 = 0; i1 < cfg.N_I_test; ++i1) {
                        tb.M_gain_I[i1] += cfg.W_r[r_idx] * tb.H_Ip[i1];
                        tb.M_gain_J[i1] += cfg.W_r[r_idx] * tb.H_Jp[i1];
                    }
                }

                // Branch B: Integrate Spatial Scattering
                std::fill(tb.S_gain1.begin(), tb.S_gain1.end(), 0.0);
                std::fill(tb.S_gain2.begin(), tb.S_gain2.end(), 0.0);
                for (int n = 0; n < cfg.N_chi; ++n) {
                    double c_chi = cfg.mu_chi[n]; 
                    double s_chi = std::sqrt(std::max(1.0 - c_chi * c_chi, 0.0));

                    for (int e_idx = 0; e_idx < cfg.N_eps; ++e_idx) {
                        double u_px = s_chi * (*cfg.c_eps)[e_idx]; 
                        double u_py = s_chi * (*cfg.s_eps)[e_idx]; 
                        double u_pz = c_chi;
                        
                        double up_x1 = u_px * z_scat_z1 + u_pz * z_scat_x1; 
                        double up_z1 = -u_px * z_scat_x1 + u_pz * z_scat_z1;
                        double vp_x1 = 0.5 * (U_x1 + u_prime * up_x1); 
                        double vp_y1 = 0.5 * (u_prime * u_py); 
                        double vp_z1 = 0.5 * (U_z1 + u_prime * up_z1);
                        double vp_mag1 = std::sqrt(std::max(vp_x1*vp_x1 + vp_y1*vp_y1 + vp_z1*vp_z1, eps_safe));
                        
                        double x_val1 = std::max(std::min(vp_z1 / vp_mag1, 1.0), -1.0);
                        double rho1 = std::sqrt(std::max(vp_x1*vp_x1 + vp_y1*vp_y1, eps_safe));
                        eval_SH_fast(x_val1, vp_x1 / rho1, vp_y1 / rho1, cfg.N_Q, cfg.SH_Norm, tb.Y_tmp1.data());
                        fill_radial_table(cfg.N_K_rad, cfg.L_max, vp_mag1, cfg.RadialNorm, tb.rad_vpmag1);
                        
                        double up_x2 = u_px * z_scat_z2 + u_pz * z_scat_x2; 
                        double up_z2 = -u_px * z_scat_x2 + u_pz * z_scat_z2;
                        double vp_x2 = 0.5 * (U_x2 + u_prime * up_x2); 
                        double vp_y2 = 0.5 * (u_prime * u_py); 
                        double vp_z2 = 0.5 * (U_z2 + u_prime * up_z2);
                        double vp_mag2 = std::sqrt(std::max(vp_x2*vp_x2 + vp_y2*vp_y2 + vp_z2*vp_z2, eps_safe));
                        
                        double x_val2 = std::max(std::min(vp_z2 / vp_mag2, 1.0), -1.0);
                        double rho2 = std::sqrt(std::max(vp_x2*vp_x2 + vp_y2*vp_y2, eps_safe));
                        eval_SH_fast(x_val2, vp_x2 / rho2, vp_y2 / rho2, cfg.N_Q, cfg.SH_Norm, tb.Y_tmp2.data());
                        fill_radial_table(cfg.N_K_rad, cfg.L_max, vp_mag2, cfg.RadialNorm, tb.rad_vpmag2);

                        for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                            int l_1 = (int)cfg.L_triplets[t_chan + cfg.N_L * 0];
                            double g_val1 = 0.0; double g_val2 = 0.0;
                            
                            for (int idx = 0; idx < cfg.N_Q; ++idx) {
                                int q_i = (int)cfg.qi_valid_mat[idx + cfg.N_Q * t_chan];
                                if (q_i == -1) break; 
                                q_i -= 1; 
                                double w_ang = P_gain[q_i * spatial_stride + cfg.N_Q * spatial_stride * t_chan];
                                g_val1 += tb.Y_tmp1[q_i] * w_ang; 
                                g_val2 += tb.Y_tmp2[q_i] * w_ang;
                            }
                            double wg1 = g_val1 * cfg.W_chi[n] * cfg.W_eps[e_idx];
                            double wg2 = g_val2 * cfg.W_chi[n] * cfg.W_eps[e_idx];
                            
                            for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                                tb.S_gain1[k1 + cfg.N_K_test * t_chan] += wg1 * tb.rad_vpmag1[k1 + cfg.N_K_rad * l_1];
                                tb.S_gain2[k1 + cfg.N_K_test * t_chan] += wg2 * tb.rad_vpmag2[k1 + cfg.N_K_rad * l_1];
                            }
                        }
                    }
                }

                // Merge Branches A and B
                for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                    for (int i1 = 0; i1 < cfg.N_I_test; ++i1) {
                        for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                            int ng_idx = k1 + cfg.N_K_test * (i1 + cfg.N_I_test * t_chan);
                            tb.N_gain1[ng_idx] += cfg.W_R[R_idx] * tb.M_gain_I[i1] * tb.S_gain1[k1 + cfg.N_K_test * t_chan];
                            tb.N_gain2[ng_idx] += cfg.W_R[R_idx] * tb.M_gain_J[i1] * tb.S_gain2[k1 + cfg.N_K_test * t_chan];
                        }
                    }
                }
            } // End R loop
            
            // Dense Contraction (Exact Particle Symmetry)
            for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                for (int i1 = 0; i1 < cfg.N_I_test; ++i1) {
                    for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                        int ng_idx = k1 + cfg.N_K_test * (i1 + cfg.N_I_test * t_chan);
                        tb.net1[k1] = factor * tb.N_gain1[ng_idx] - tb.Phi_loss1[ng_idx];
                        tb.net2[k1] = factor * tb.N_gain2[ng_idx] - tb.Phi_loss2[ng_idx];
                    }
                    
                    int l_2 = (int)cfg.L_triplets[t_chan + cfg.N_L * 1]; 
                    int l_3 = (int)cfg.L_triplets[t_chan + cfg.N_L * 2];
                    
                    for (int i3 = 0; i3 < cfg.N_I; ++i3) {
                        for (int i2 = 0; i2 < cfg.N_I; ++i2) {
                            for (int k3 = 0; k3 < cfg.N_K; ++k3) {
                                for (int k2 = 0; k2 < cfg.N_K; ++k2) {
                                    double p1 = tb.H_I_trial[i2] * tb.H_J_trial[i3] * tb.Rad_large[k2 + cfg.N_K_rad * l_2] * tb.Rad_small[k3 + cfg.N_K_rad * l_3];
                                    double p2 = tb.H_J_trial[i2] * tb.H_I_trial[i3] * tb.Rad_small[k2 + cfg.N_K_rad * l_2] * tb.Rad_large[k3 + cfg.N_K_rad * l_3];
                                    
                                    int out_idx = 0 + cfg.N_K_test * (k2 + cfg.N_K * (k3 + cfg.N_K * (i1 + cfg.N_I_test * (i2 + cfg.N_I * (i3 + cfg.N_I * t_chan)))));
                                    
                                    double* R_ptr = &tb.R_local[out_idx];
                                    #pragma omp simd
                                    for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                                        R_ptr[k1] += tb.net1[k1] * p1 + tb.net2[k1] * p2;
                                    }
                                }
                            }
                        }
                    }
                }
            } // End dense contraction
        } // End J loop
    } // End I loop
}

// ========================================================================
// PATCH EVALUATIONS (SPATIAL WRAPPERS)
// ========================================================================

inline void compute_patch1(const Config& cfg, int i, double x_i, ThreadScratch& tb) {
    const double eps_safe = 1e-15;

    for (int u_idx = 0; u_idx < cfg.N_u1; ++u_idx) {
        double u = cfg.u1_nodes[u_idx];
        double v_large = std::sqrt(x_i / 2.0) * (1.0 + u);
        double v_small = std::sqrt(x_i / 2.0) * (1.0 - u);
        
        fill_radial_table(cfg.N_K_rad, cfg.L_max, v_large, cfg.RadialNorm, tb.Rad_large);
        fill_radial_table(cfg.N_K_rad, cfg.L_max, v_small, cfg.RadialNorm, tb.Rad_small);

        for (int t_idx = 0; t_idx < cfg.N_t1; ++t_idx) {
            double t = cfg.t1_nodes[t_idx]; double y = u * t;
            double c_beta = 1.0 - 2.0 * y * y;
            double s_beta = std::sqrt(std::max(1.0 - c_beta * c_beta, eps_safe));
            
            double Jac_spatial = 0.5 * cfg.W_x[i] * cfg.W_u1[u_idx] * u * std::exp(-x_i * u * u) * (v_large * v_large * v_small * v_small) * cfg.W_t1[t_idx] * (4.0 * u * t);
            double u_mag = std::sqrt(2.0 * x_i) * u * std::sqrt(1.0 + (1.0 - u * u) * t * t);

            double U_x1 = v_small * s_beta; double U_z1 = v_large + v_small * c_beta;
            double u_x1 = -v_small * s_beta; double u_z1 = v_large - v_small * c_beta;
            
            double U_x2 = v_large * s_beta; double U_z2 = v_small + v_large * c_beta;
            double u_x2 = -v_large * s_beta; double u_z2 = v_small - v_large * c_beta;

            int spatial_idx = t_idx + cfg.N_t1 * u_idx;
            int spatial_stride = cfg.N_t1 * cfg.N_u1;
            const double* P_loss_ptr = &cfg.P_loss_p1[spatial_idx];
            const double* P_gain_ptr = &cfg.P_gain_p1[spatial_idx];

            compute_polyatomic_inner(cfg, tb, Jac_spatial, u_mag, x_i,
                                     U_x1, U_z1, u_x1, u_z1, U_x2, U_z2, u_x2, u_z2, 
                                     P_loss_ptr, P_gain_ptr, spatial_stride);
        }
    }
}

inline void compute_patch2(const Config& cfg, int i, double x_i, ThreadScratch& tb) {
    const double eps_safe = 1e-15;

    for (int y_idx = 0; y_idx < cfg.N_y2; ++y_idx) {
        double y = cfg.y2_nodes[y_idx];
        double c_beta = 1.0 - 2.0 * y * y;
        double s_beta = std::sqrt(std::max(1.0 - c_beta * c_beta, eps_safe));

        for (int t_idx = 0; t_idx < cfg.N_t2; ++t_idx) {
            double t = cfg.t2_nodes[t_idx]; double u = y * t;
            
            double v_large = std::sqrt(x_i / 2.0) * (1.0 + u);
            double v_small = std::sqrt(x_i / 2.0) * (1.0 - u);
            
            fill_radial_table(cfg.N_K_rad, cfg.L_max, v_large, cfg.RadialNorm, tb.Rad_large);
            fill_radial_table(cfg.N_K_rad, cfg.L_max, v_small, cfg.RadialNorm, tb.Rad_small);
            
            double Jac_spatial = 0.5 * cfg.W_x[i] * cfg.W_y2[y_idx] * y * (4.0 * y) * cfg.W_t2[t_idx] * std::exp(-x_i * u * u) * (v_large * v_large * v_small * v_small);
            double u_mag = std::sqrt(2.0 * x_i) * y * std::sqrt(t * t + 1.0 - y * y * t * t);

            double U_x1 = v_small * s_beta; double U_z1 = v_large + v_small * c_beta;
            double u_x1 = -v_small * s_beta; double u_z1 = v_large - v_small * c_beta;
            
            double U_x2 = v_large * s_beta; double U_z2 = v_small + v_large * c_beta;
            double u_x2 = -v_large * s_beta; double u_z2 = v_small - v_large * c_beta;

            int spatial_idx = y_idx;
            int spatial_stride = cfg.N_y2;
            const double* P_loss_ptr = &cfg.P_loss_p2[spatial_idx];
            const double* P_gain_ptr = &cfg.P_gain_p2[spatial_idx];

            compute_polyatomic_inner(cfg, tb, Jac_spatial, u_mag, x_i,
                                     U_x1, U_z1, u_x1, u_z1, U_x2, U_z2, u_x2, u_z2, 
                                     P_loss_ptr, P_gain_ptr, spatial_stride);
        }
    }
}

// ========================================================================
// MEX MAIN FUNCTION
// ========================================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 39) {
        mexErrMsgIdAndTxt("R_tensor_polyatomic:InvalidInput", "Requires exactly 39 inputs for Polyatomic Sum-Factorization.");
    }
    
    Config cfg;
    cfg.K_max = (int)mxGetScalar(prhs[0]); cfg.N_K = cfg.K_max + 1;
    cfg.I_max = (int)mxGetScalar(prhs[1]); cfg.N_I = cfg.I_max + 1;
    cfg.N_L = (int)mxGetScalar(prhs[2]);
    cfg.N_Q = (int)mxGetScalar(prhs[3]);
    cfg.alpha = mxGetScalar(prhs[4]);
    cfg.nu = mxGetScalar(prhs[5]);
    cfg.L_max = (int)std::round(std::sqrt(cfg.N_Q)) - 1;

    cfg.x_nodes = mxGetPr(prhs[6]); cfg.W_x = mxGetPr(prhs[7]); cfg.N_x = mxGetNumberOfElements(prhs[6]);
    cfg.u1_nodes = mxGetPr(prhs[8]); cfg.W_u1 = mxGetPr(prhs[9]); cfg.N_u1 = mxGetNumberOfElements(prhs[8]);
    cfg.t1_nodes = mxGetPr(prhs[10]); cfg.W_t1 = mxGetPr(prhs[11]); cfg.N_t1 = mxGetNumberOfElements(prhs[10]);
    cfg.y2_nodes = mxGetPr(prhs[12]); cfg.W_y2 = mxGetPr(prhs[13]); cfg.N_y2 = mxGetNumberOfElements(prhs[12]);
    cfg.t2_nodes = mxGetPr(prhs[14]); cfg.W_t2 = mxGetPr(prhs[15]); cfg.N_t2 = mxGetNumberOfElements(prhs[14]);
    cfg.mu_chi = mxGetPr(prhs[16]); cfg.W_chi = mxGetPr(prhs[17]); cfg.N_chi = mxGetNumberOfElements(prhs[16]);
    cfg.eps_vec = mxGetPr(prhs[18]); cfg.W_eps = mxGetPr(prhs[19]); cfg.N_eps = mxGetNumberOfElements(prhs[18]);
    
    cfg.I_nodes = mxGetPr(prhs[20]); cfg.W_I = mxGetPr(prhs[21]); cfg.N_I_nodes = mxGetNumberOfElements(prhs[20]);
    cfg.J_nodes = mxGetPr(prhs[22]); cfg.W_J = mxGetPr(prhs[23]); cfg.N_J_nodes = mxGetNumberOfElements(prhs[22]);
    cfg.r_nodes = mxGetPr(prhs[24]); cfg.W_r = mxGetPr(prhs[25]); cfg.N_r_nodes = mxGetNumberOfElements(prhs[24]);
    cfg.R_nodes = mxGetPr(prhs[26]); cfg.W_R = mxGetPr(prhs[27]); cfg.N_R_nodes = mxGetNumberOfElements(prhs[26]);
    
    cfg.RadialNorm = mxGetPr(prhs[28]); cfg.N_K_rad = (int)mxGetM(prhs[28]); 
    cfg.InternalNorm = mxGetPr(prhs[29]); cfg.N_I_rad = (int)mxGetM(prhs[29]);
    
    cfg.SH_Norm = mxGetPr(prhs[30]);    
    cfg.L_triplets = mxGetPr(prhs[31]); 
    cfg.qi_valid_mat = mxGetPr(prhs[32]); 
    
    cfg.P_loss_p1 = mxGetPr(prhs[33]); 
    cfg.P_gain_p1 = mxGetPr(prhs[34]); 
    cfg.P_loss_p2 = mxGetPr(prhs[35]); 
    cfg.P_gain_p2 = mxGetPr(prhs[36]); 

    cfg.K_test = (int)mxGetScalar(prhs[37]); cfg.N_K_test = cfg.K_test + 1;
    cfg.I_test = (int)mxGetScalar(prhs[38]); cfg.N_I_test = cfg.I_test + 1;

    cfg.sum_WR = 0.0; for(int q=0; q<cfg.N_R_nodes; ++q) cfg.sum_WR += cfg.W_R[q];
    cfg.sum_Wr = 0.0; for(int q=0; q<cfg.N_r_nodes; ++q) cfg.sum_Wr += cfg.W_r[q];

    mwSize dims[7] = {(mwSize)cfg.N_K_test, (mwSize)cfg.N_K, (mwSize)cfg.N_K, 
                      (mwSize)cfg.N_I_test, (mwSize)cfg.N_I, (mwSize)cfg.N_I, 
                      (mwSize)cfg.N_L};
    plhs[0] = mxCreateNumericArray(7, dims, mxDOUBLE_CLASS, mxREAL);
    double* R_tensor = mxGetPr(plhs[0]);

    std::vector<double> c_eps(cfg.N_eps, 0.0);
    std::vector<double> s_eps(cfg.N_eps, 0.0);
    for (int e = 0; e < cfg.N_eps; ++e) {
        c_eps[e] = std::cos(cfg.eps_vec[e]);
        s_eps[e] = std::sin(cfg.eps_vec[e]);
    }
    cfg.c_eps = &c_eps;
    cfg.s_eps = &s_eps;

    #pragma omp parallel
    {
        ThreadScratch tb(cfg);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < cfg.N_x; ++i) {
            double x_i = cfg.x_nodes[i];
            compute_patch1(cfg, i, x_i, tb);
            compute_patch2(cfg, i, x_i, tb);
        }

        #pragma omp critical
        {
            int total_elements = cfg.N_K_test * cfg.N_K * cfg.N_K * cfg.N_I_test * cfg.N_I * cfg.N_I * cfg.N_L;
            for (int i = 0; i < total_elements; ++i) {
                R_tensor[i] += tb.R_local[i];
            }
        }
    }
}