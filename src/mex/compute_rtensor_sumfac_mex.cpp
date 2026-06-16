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
            double L0 = 1.0;
            double L1 = 1.0 + al - v2;
            for (int k = 2; k < N_K_rad; ++k) {
                int i = k - 1;
                double Lk = ((2.0 * i + 1.0 + al - v2) * L1 - (i + al) * L0) / (i + 1.0);
                table[k + N_K_rad * l] = RadialNorm[k + N_K_rad * l] * Lk * vl;
                L0 = L1;
                L1 = Lk;
            }
        }
    }
}

// Trig-Free Spherical Harmonic Evaluator
inline void eval_SH_fast(double x, double cphi, double sphi, int N_Q, const double* SH_Norm, double* Y_out) {
    int L_max = (int)std::round(std::sqrt(N_Q)) - 1;
    double somx2 = std::sqrt(std::max((1.0 - x) * (1.0 + x), 0.0));

    // Chebyshev recurrence for cos(m*phi) and sin(m*phi)
    double cosm[256]; 
    double sinm[256];
    cosm[0] = 1.0; sinm[0] = 0.0;
    if (L_max >= 1) {
        cosm[1] = cphi; sinm[1] = sphi;
    }
    for (int m = 2; m <= L_max; ++m) {
        cosm[m] = 2.0 * cphi * cosm[m - 1] - cosm[m - 2];
        sinm[m] = 2.0 * cphi * sinm[m - 1] - sinm[m - 2];
    }

    double pmm = 1.0;
    for (int m = 0; m <= L_max; ++m) {
        if (m > 0) pmm *= -(2.0 * m - 1.0) * somx2;

        double Plm2, Plm1, Pl;
        int base_mm = m * m + m;
        
        if (m == 0) {
            Y_out[base_mm] = SH_Norm[base_mm] * pmm;
        } else {
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
    int K_max, N_K, N_K_test, N_K_rad, N_L, N_Q, L_max;
    double alpha;
    const double *x_nodes, *W_x; int N_x;
    const double *u1_nodes, *W_u1; int N_u1;
    const double *t1_nodes, *W_t1; int N_t1;
    const double *y2_nodes, *W_y2; int N_y2;
    const double *t2_nodes, *W_t2; int N_t2;
    const double *mu_chi, *W_chi; int N_chi;
    const double *eps_vec, *W_eps; int N_eps;
    
    const double *RadialNorm, *SH_Norm, *L_triplets, *qi_valid_mat;
    const double *P_loss_p1, *P_gain_p1, *P_loss_p2, *P_gain_p2;
    
    const std::vector<double>* c_eps;
    const std::vector<double>* s_eps;
};

struct ThreadScratch {
    std::vector<double> R_local;
    std::vector<double> Y_tmp1;
    std::vector<double> Y_tmp2;
    std::vector<double> Phi_loss1;
    std::vector<double> Phi_loss2;
    std::vector<double> Phi_gain1;
    std::vector<double> Phi_gain2;
    std::vector<double> Rad_large;
    std::vector<double> Rad_small;
    std::vector<double> net1;
    std::vector<double> net2;

    ThreadScratch(const Config& cfg) {
        R_local.assign(cfg.N_K_test * cfg.N_K * cfg.N_K * cfg.N_L, 0.0);
        Y_tmp1.assign(cfg.N_Q, 0.0);
        Y_tmp2.assign(cfg.N_Q, 0.0);
        
        // Accumulators are sized to the test space (k1)
        Phi_loss1.assign(cfg.N_K_test * cfg.N_L, 0.0);
        Phi_loss2.assign(cfg.N_K_test * cfg.N_L, 0.0);
        Phi_gain1.assign(cfg.N_K_test * cfg.N_L, 0.0);
        Phi_gain2.assign(cfg.N_K_test * cfg.N_L, 0.0);
        
        // Radial tables scaled to max required dimension
        Rad_large.assign(cfg.N_K_rad * (cfg.L_max + 1), 0.0);
        Rad_small.assign(cfg.N_K_rad * (cfg.L_max + 1), 0.0);
        
        net1.assign(cfg.N_K_test, 0.0);
        net2.assign(cfg.N_K_test, 0.0);
    }
};

// ========================================================================
// PATCH EVALUATIONS
// ========================================================================

inline void compute_patch1(const Config& cfg, int i, double x_i, ThreadScratch& tb) {
    const double eps_safe = 1e-15;

    for (int u_idx = 0; u_idx < cfg.N_u1; ++u_idx) {
        double u = cfg.u1_nodes[u_idx];
        double v_large = std::sqrt(x_i / 2.0) * (1.0 + u);
        double v_small = std::sqrt(x_i / 2.0) * (1.0 - u);
        
        double Jac_u = 0.5 * cfg.W_x[i] * cfg.W_u1[u_idx] * u * std::exp(-x_i * u * u) * (v_large * v_large * v_small * v_small);
        
        fill_radial_table(cfg.N_K_rad, cfg.L_max, v_large, cfg.RadialNorm, tb.Rad_large);
        fill_radial_table(cfg.N_K_rad, cfg.L_max, v_small, cfg.RadialNorm, tb.Rad_small);

        std::fill(tb.Phi_loss1.begin(), tb.Phi_loss1.end(), 0.0); 
        std::fill(tb.Phi_loss2.begin(), tb.Phi_loss2.end(), 0.0);
        std::fill(tb.Phi_gain1.begin(), tb.Phi_gain1.end(), 0.0); 
        std::fill(tb.Phi_gain2.begin(), tb.Phi_gain2.end(), 0.0);

        for (int t_idx = 0; t_idx < cfg.N_t1; ++t_idx) {
            double t = cfg.t1_nodes[t_idx]; 
            double y = u * t;
            double Jac_t = cfg.W_t1[t_idx] * (4.0 * u * t);
            
            double c_beta = 1.0 - 2.0 * y * y;
            double s_beta = std::sqrt(std::max(1.0 - c_beta * c_beta, eps_safe));
            
            double u_mag = std::sqrt(2.0 * x_i) * u * std::sqrt(1.0 + (1.0 - u * u) * t * t);
            double B_val = std::pow(u_mag * std::sqrt(2.0), cfg.alpha) / std::pow(x_i, cfg.alpha / 2.0);
            double loss_weight = B_val * 2.0 * M_PI * 2.0 * Jac_t;

            for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                int l_1 = (int)cfg.L_triplets[t_chan + cfg.N_L * 0];
                double loss_int = loss_weight * cfg.P_loss_p1[t_idx + cfg.N_t1 * (u_idx + cfg.N_u1 * t_chan)];
                
                // Loss integration strictly for test basis (up to N_K_test)
                for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                    tb.Phi_loss1[k1 + cfg.N_K_test * t_chan] += tb.Rad_large[k1 + cfg.N_K_rad * l_1] * loss_int;
                    tb.Phi_loss2[k1 + cfg.N_K_test * t_chan] += tb.Rad_small[k1 + cfg.N_K_rad * l_1] * loss_int;
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

            for (int n = 0; n < cfg.N_chi; ++n) {
                double c_chi = cfg.mu_chi[n]; 
                double s_chi = std::sqrt(std::max(1.0 - c_chi * c_chi, 0.0));
                double gain_base = cfg.W_chi[n] * B_val * Jac_t;

                for (int e_idx = 0; e_idx < cfg.N_eps; ++e_idx) {
                    double u_px = s_chi * (*cfg.c_eps)[e_idx]; 
                    double u_py = s_chi * (*cfg.s_eps)[e_idx]; 
                    double u_pz = c_chi;
                    
                    double up_x1 = u_px * z_scat_z1 + u_pz * z_scat_x1; 
                    double up_z1 = -u_px * z_scat_x1 + u_pz * z_scat_z1;
                    double vp_x1 = 0.5 * (U_x1 + u_mag1 * up_x1); 
                    double vp_y1 = 0.5 * (u_mag1 * u_py); 
                    double vp_z1 = 0.5 * (U_z1 + u_mag1 * up_z1);
                    double vp_mag1 = std::sqrt(std::max(vp_x1*vp_x1 + vp_y1*vp_y1 + vp_z1*vp_z1, eps_safe));
                    
                    double x_val1 = std::max(std::min(vp_z1 / vp_mag1, 1.0), -1.0);
                    double rho1 = std::sqrt(std::max(vp_x1*vp_x1 + vp_y1*vp_y1, eps_safe));
                    eval_SH_fast(x_val1, vp_x1 / rho1, vp_y1 / rho1, cfg.N_Q, cfg.SH_Norm, tb.Y_tmp1.data());
                    
                    double up_x2 = u_px * z_scat_z2 + u_pz * z_scat_x2; 
                    double up_z2 = -u_px * z_scat_x2 + u_pz * z_scat_z2;
                    double vp_x2 = 0.5 * (U_x2 + u_mag2 * up_x2); 
                    double vp_y2 = 0.5 * (u_mag2 * u_py); 
                    double vp_z2 = 0.5 * (U_z2 + u_mag2 * up_z2);
                    double vp_mag2 = std::sqrt(std::max(vp_x2*vp_x2 + vp_y2*vp_y2 + vp_z2*vp_z2, eps_safe));
                    
                    double x_val2 = std::max(std::min(vp_z2 / vp_mag2, 1.0), -1.0);
                    double rho2 = std::sqrt(std::max(vp_x2*vp_x2 + vp_y2*vp_y2, eps_safe));
                    eval_SH_fast(x_val2, vp_x2 / rho2, vp_y2 / rho2, cfg.N_Q, cfg.SH_Norm, tb.Y_tmp2.data());

                    // Evaluate target particle's post-collision radial basis inside azimuthal loop
                    std::vector<double> rad_vpmag1(cfg.N_K_rad * (cfg.L_max + 1), 0.0);
                    std::vector<double> rad_vpmag2(cfg.N_K_rad * (cfg.L_max + 1), 0.0);
                    fill_radial_table(cfg.N_K_rad, cfg.L_max, vp_mag1, cfg.RadialNorm, rad_vpmag1);
                    fill_radial_table(cfg.N_K_rad, cfg.L_max, vp_mag2, cfg.RadialNorm, rad_vpmag2);

                    for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                        int l_1 = (int)cfg.L_triplets[t_chan + cfg.N_L * 0];
                        double g_val1 = 0.0; double g_val2 = 0.0;
                        
                        for (int idx = 0; idx < cfg.N_Q; ++idx) {
                            int q_i = (int)cfg.qi_valid_mat[idx + cfg.N_Q * t_chan];
                            if (q_i == -1) break; 
                            q_i -= 1; 
                            
                            double w_ang = cfg.P_gain_p1[t_idx + cfg.N_t1 * (u_idx + cfg.N_u1 * (q_i + cfg.N_Q * t_chan))];
                            g_val1 += tb.Y_tmp1[q_i] * w_ang; 
                            g_val2 += tb.Y_tmp2[q_i] * w_ang;
                        }
                        double wg1 = g_val1 * gain_base * cfg.W_eps[e_idx];
                        double wg2 = g_val2 * gain_base * cfg.W_eps[e_idx];
                        
                        // Gain integration strictly for test basis (up to N_K_test)
                        for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                            tb.Phi_gain1[k1 + cfg.N_K_test * t_chan] += wg1 * rad_vpmag1[k1 + cfg.N_K_rad * l_1];
                            tb.Phi_gain2[k1 + cfg.N_K_test * t_chan] += wg2 * rad_vpmag2[k1 + cfg.N_K_rad * l_1];
                        }
                    }
                }
            }
        } // End Inner t-loop

        for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
            int l_2 = (int)cfg.L_triplets[t_chan + cfg.N_L * 1]; 
            int l_3 = (int)cfg.L_triplets[t_chan + cfg.N_L * 2];
            
            for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                tb.net1[k1] = Jac_u * (tb.Phi_gain1[k1 + cfg.N_K_test * t_chan] - tb.Phi_loss1[k1 + cfg.N_K_test * t_chan]);
                tb.net2[k1] = Jac_u * (tb.Phi_gain2[k1 + cfg.N_K_test * t_chan] - tb.Phi_loss2[k1 + cfg.N_K_test * t_chan]);
            }
            
            for (int k3 = 0; k3 < cfg.N_K; ++k3) {
                double r3_l = tb.Rad_large[k3 + cfg.N_K_rad * l_3];
                double r3_s = tb.Rad_small[k3 + cfg.N_K_rad * l_3];
                for (int k2 = 0; k2 < cfg.N_K; ++k2) {
                    double r2_l = tb.Rad_large[k2 + cfg.N_K_rad * l_2];
                    double r2_s = tb.Rad_small[k2 + cfg.N_K_rad * l_2];
                    
                    double p1 = r2_l * r3_s;
                    double p2 = r2_s * r3_l;
                    
                    // Asymmetric Strides for Tensor Memory Layout
                    double* R_ptr = &tb.R_local[0 + cfg.N_K_test * (k2 + cfg.N_K * (k3 + cfg.N_K * t_chan))];
                    
                    #pragma omp simd
                    for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                        R_ptr[k1] += tb.net1[k1] * p1 + tb.net2[k1] * p2;
                    }
                }
            }
        }
    } // End Patch 1 outer loop
}

inline void compute_patch2(const Config& cfg, int i, double x_i, ThreadScratch& tb) {
    const double eps_safe = 1e-15;

    for (int y_idx = 0; y_idx < cfg.N_y2; ++y_idx) {
        double y = cfg.y2_nodes[y_idx];
        double Jac_y = 0.5 * cfg.W_x[i] * cfg.W_y2[y_idx] * y * (4.0 * y);
        
        double c_beta = 1.0 - 2.0 * y * y;
        double s_beta = std::sqrt(std::max(1.0 - c_beta * c_beta, eps_safe));

        for (int t_idx = 0; t_idx < cfg.N_t2; ++t_idx) {
            double t = cfg.t2_nodes[t_idx]; 
            double u = y * t;
            
            double v_large = std::sqrt(x_i / 2.0) * (1.0 + u);
            double v_small = std::sqrt(x_i / 2.0) * (1.0 - u);
            
            fill_radial_table(cfg.N_K_rad, cfg.L_max, v_large, cfg.RadialNorm, tb.Rad_large);
            fill_radial_table(cfg.N_K_rad, cfg.L_max, v_small, cfg.RadialNorm, tb.Rad_small);
            
            double Jac_t = cfg.W_t2[t_idx] * std::exp(-x_i * u * u) * (v_large * v_large * v_small * v_small);
            double u_mag = std::sqrt(2.0 * x_i) * y * std::sqrt(t * t + 1.0 - y * y * t * t);
            double B_val = std::pow(u_mag * std::sqrt(2.0), cfg.alpha) / std::pow(x_i, cfg.alpha / 2.0);
            double loss_weight = B_val * 2.0 * M_PI * 2.0 * Jac_t;

            std::fill(tb.Phi_loss1.begin(), tb.Phi_loss1.end(), 0.0); 
            std::fill(tb.Phi_loss2.begin(), tb.Phi_loss2.end(), 0.0);
            std::fill(tb.Phi_gain1.begin(), tb.Phi_gain1.end(), 0.0); 
            std::fill(tb.Phi_gain2.begin(), tb.Phi_gain2.end(), 0.0);

            for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                int l_1 = (int)cfg.L_triplets[t_chan + cfg.N_L * 0];
                double loss_int = loss_weight * cfg.P_loss_p2[y_idx + cfg.N_y2 * t_chan];
                
                for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                    tb.Phi_loss1[k1 + cfg.N_K_test * t_chan] += tb.Rad_large[k1 + cfg.N_K_rad * l_1] * loss_int;
                    tb.Phi_loss2[k1 + cfg.N_K_test * t_chan] += tb.Rad_small[k1 + cfg.N_K_rad * l_1] * loss_int;
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

            for (int n = 0; n < cfg.N_chi; ++n) {
                double c_chi = cfg.mu_chi[n]; 
                double s_chi = std::sqrt(std::max(1.0 - c_chi * c_chi, 0.0));
                double gain_base = cfg.W_chi[n] * B_val * Jac_t;

                for (int e_idx = 0; e_idx < cfg.N_eps; ++e_idx) {
                    double u_px = s_chi * (*cfg.c_eps)[e_idx]; 
                    double u_py = s_chi * (*cfg.s_eps)[e_idx]; 
                    double u_pz = c_chi;
                    
                    double up_x1 = u_px * z_scat_z1 + u_pz * z_scat_x1; 
                    double up_z1 = -u_px * z_scat_x1 + u_pz * z_scat_z1;
                    double vp_x1 = 0.5 * (U_x1 + u_mag1 * up_x1); 
                    double vp_y1 = 0.5 * (u_mag1 * u_py); 
                    double vp_z1 = 0.5 * (U_z1 + u_mag1 * up_z1);
                    double vp_mag1 = std::sqrt(std::max(vp_x1*vp_x1 + vp_y1*vp_y1 + vp_z1*vp_z1, eps_safe));
                    
                    double x_val1 = std::max(std::min(vp_z1 / vp_mag1, 1.0), -1.0);
                    double rho1 = std::sqrt(std::max(vp_x1*vp_x1 + vp_y1*vp_y1, eps_safe));
                    eval_SH_fast(x_val1, vp_x1 / rho1, vp_y1 / rho1, cfg.N_Q, cfg.SH_Norm, tb.Y_tmp1.data());
                    
                    double up_x2 = u_px * z_scat_z2 + u_pz * z_scat_x2; 
                    double up_z2 = -u_px * z_scat_x2 + u_pz * z_scat_z2;
                    double vp_x2 = 0.5 * (U_x2 + u_mag2 * up_x2); 
                    double vp_y2 = 0.5 * (u_mag2 * u_py); 
                    double vp_z2 = 0.5 * (U_z2 + u_mag2 * up_z2);
                    double vp_mag2 = std::sqrt(std::max(vp_x2*vp_x2 + vp_y2*vp_y2 + vp_z2*vp_z2, eps_safe));
                    
                    double x_val2 = std::max(std::min(vp_z2 / vp_mag2, 1.0), -1.0);
                    double rho2 = std::sqrt(std::max(vp_x2*vp_x2 + vp_y2*vp_y2, eps_safe));
                    eval_SH_fast(x_val2, vp_x2 / rho2, vp_y2 / rho2, cfg.N_Q, cfg.SH_Norm, tb.Y_tmp2.data());

                    std::vector<double> rad_vpmag1(cfg.N_K_rad * (cfg.L_max + 1), 0.0);
                    std::vector<double> rad_vpmag2(cfg.N_K_rad * (cfg.L_max + 1), 0.0);
                    fill_radial_table(cfg.N_K_rad, cfg.L_max, vp_mag1, cfg.RadialNorm, rad_vpmag1);
                    fill_radial_table(cfg.N_K_rad, cfg.L_max, vp_mag2, cfg.RadialNorm, rad_vpmag2);

                    for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                        int l_1 = (int)cfg.L_triplets[t_chan + cfg.N_L * 0];
                        double g_val1 = 0.0; double g_val2 = 0.0;
                        
                        for (int idx = 0; idx < cfg.N_Q; ++idx) {
                            int q_i = (int)cfg.qi_valid_mat[idx + cfg.N_Q * t_chan];
                            if (q_i == -1) break; 
                            q_i -= 1; 
                            
                            double w_ang = cfg.P_gain_p2[y_idx + cfg.N_y2 * (q_i + cfg.N_Q * t_chan)];
                            g_val1 += tb.Y_tmp1[q_i] * w_ang; 
                            g_val2 += tb.Y_tmp2[q_i] * w_ang;
                        }
                        double wg1 = g_val1 * gain_base * cfg.W_eps[e_idx];
                        double wg2 = g_val2 * gain_base * cfg.W_eps[e_idx];
                        
                        for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                            tb.Phi_gain1[k1 + cfg.N_K_test * t_chan] += wg1 * rad_vpmag1[k1 + cfg.N_K_rad * l_1];
                            tb.Phi_gain2[k1 + cfg.N_K_test * t_chan] += wg2 * rad_vpmag2[k1 + cfg.N_K_rad * l_1];
                        }
                    }
                }
            }

            for (int t_chan = 0; t_chan < cfg.N_L; ++t_chan) {
                int l_2 = (int)cfg.L_triplets[t_chan + cfg.N_L * 1]; 
                int l_3 = (int)cfg.L_triplets[t_chan + cfg.N_L * 2];
                
                for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                    tb.net1[k1] = Jac_y * (tb.Phi_gain1[k1 + cfg.N_K_test * t_chan] - tb.Phi_loss1[k1 + cfg.N_K_test * t_chan]);
                    tb.net2[k1] = Jac_y * (tb.Phi_gain2[k1 + cfg.N_K_test * t_chan] - tb.Phi_loss2[k1 + cfg.N_K_test * t_chan]);
                }
                
                for (int k3 = 0; k3 < cfg.N_K; ++k3) {
                    double r3_l = tb.Rad_large[k3 + cfg.N_K_rad * l_3];
                    double r3_s = tb.Rad_small[k3 + cfg.N_K_rad * l_3];
                    for (int k2 = 0; k2 < cfg.N_K; ++k2) {
                        double r2_l = tb.Rad_large[k2 + cfg.N_K_rad * l_2];
                        double r2_s = tb.Rad_small[k2 + cfg.N_K_rad * l_2];
                        
                        double p1 = r2_l * r3_s;
                        double p2 = r2_s * r3_l;
                        
                        double* R_ptr = &tb.R_local[0 + cfg.N_K_test * (k2 + cfg.N_K * (k3 + cfg.N_K * t_chan))];
                        
                        #pragma omp simd
                        for (int k1 = 0; k1 < cfg.N_K_test; ++k1) {
                            R_ptr[k1] += tb.net1[k1] * p1 + tb.net2[k1] * p2;
                        }
                    }
                }
            }
        } // End Inner t-loop
    } // End Patch 2 outer loop
}

// ========================================================================
// MEX MAIN FUNCTION
// ========================================================================
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    if (nrhs != 26 && nrhs != 27) {
        mexErrMsgIdAndTxt("R_tensor:InvalidInput", "Requires 26 or 27 inputs.");
    }
    
    Config cfg;
    cfg.K_max = (int)mxGetScalar(prhs[0]); 
    cfg.N_K = cfg.K_max + 1;
    cfg.N_L = (int)mxGetScalar(prhs[1]);
    cfg.N_Q = (int)mxGetScalar(prhs[2]);
    cfg.alpha = mxGetScalar(prhs[3]);
    cfg.L_max = (int)std::round(std::sqrt(cfg.N_Q)) - 1;

    int K_test = cfg.K_max;
    if (nrhs == 27) {
        K_test = (int)mxGetScalar(prhs[26]);
    }
    cfg.N_K_test = K_test + 1;
    
    cfg.x_nodes = mxGetPr(prhs[4]); cfg.W_x = mxGetPr(prhs[5]); cfg.N_x = mxGetNumberOfElements(prhs[4]);
    cfg.u1_nodes = mxGetPr(prhs[6]); cfg.W_u1 = mxGetPr(prhs[7]); cfg.N_u1 = mxGetNumberOfElements(prhs[6]);
    cfg.t1_nodes = mxGetPr(prhs[8]); cfg.W_t1 = mxGetPr(prhs[9]); cfg.N_t1 = mxGetNumberOfElements(prhs[8]);
    cfg.y2_nodes = mxGetPr(prhs[10]); cfg.W_y2 = mxGetPr(prhs[11]); cfg.N_y2 = mxGetNumberOfElements(prhs[10]);
    cfg.t2_nodes = mxGetPr(prhs[12]); cfg.W_t2 = mxGetPr(prhs[13]); cfg.N_t2 = mxGetNumberOfElements(prhs[12]);
    cfg.mu_chi = mxGetPr(prhs[14]); cfg.W_chi = mxGetPr(prhs[15]); cfg.N_chi = mxGetNumberOfElements(prhs[14]);
    cfg.eps_vec = mxGetPr(prhs[16]); cfg.W_eps = mxGetPr(prhs[17]); cfg.N_eps = mxGetNumberOfElements(prhs[16]);
    
    cfg.RadialNorm = mxGetPr(prhs[18]); 
    cfg.N_K_rad = (int)mxGetM(prhs[18]); // Dynamically fetch actual padded basis size
    
    cfg.SH_Norm = mxGetPr(prhs[19]);    
    cfg.L_triplets = mxGetPr(prhs[20]); 
    cfg.qi_valid_mat = mxGetPr(prhs[21]); 
    
    cfg.P_loss_p1 = mxGetPr(prhs[22]); 
    cfg.P_gain_p1 = mxGetPr(prhs[23]); 
    cfg.P_loss_p2 = mxGetPr(prhs[24]); 
    cfg.P_gain_p2 = mxGetPr(prhs[25]); 

    mwSize dims[4] = {(mwSize)cfg.N_K_test, (mwSize)cfg.N_K, (mwSize)cfg.N_K, (mwSize)cfg.N_L};
    plhs[0] = mxCreateNumericArray(4, dims, mxDOUBLE_CLASS, mxREAL);
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
            for (int i = 0; i < cfg.N_K_test * cfg.N_K * cfg.N_K * cfg.N_L; ++i) {
                R_tensor[i] += tb.R_local[i];
            }
        }
    }
}