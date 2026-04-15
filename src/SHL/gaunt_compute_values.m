function [labels, values] = gaunt_compute_values(L_max)
    % Pre-compute log-factorials for machine-precision stability
    max_fact = 3 * L_max + 2;
    lf = [0, cumsum(log(1:max_fact))]; % lf(n+1) is log(n!)

    % Generate the skeleton
    labels = gaunt_generate_labels(L_max);
    num_nz = size(labels, 1);
    values = zeros(num_nz, 1);
    
    % Helper to map q -> (l, m)
    q2l = @(q) floor(sqrt(double(q)-1));
    q2m = @(q, l) (double(q) - 1) - l.^2 - l;

    fprintf('Computing %d Complex Gaunt coefficients...\n', num_nz);

    % Map all indices to l once
    all_l = q2l(labels);
    [unique_triplets, ~, ic] = unique(all_l, 'rows');
    
    for i = 1:size(unique_triplets, 1)
        l_trip = unique_triplets(i, :);
        l1 = l_trip(1); l2 = l_trip(2); l3 = l_trip(3);
        
        % 1. Compute the "0-0-0" part (Constant for this l-triplet)
        w0 = fast_w3j_zero(l1, l2, l3, lf);
        if w0 == 0, continue; end % Should be caught by parity, but for safety
        
        % 2. Find all labels belonging to this l-triplet
        mask = (ic == i);
        sub_labels = labels(mask, :);
        
        % 3. Extract m-values
        m1 = q2m(sub_labels(:,1), l1);
        m2 = q2m(sub_labels(:,2), l2);
        m3 = q2m(sub_labels(:,3), l3);
        
        % 4. Vectorized Wigner-3j (Magnetic part)
        % Gaunt includes (-1)^m3 because it's Y1*Y2*Y3_star
        pref = (-1).^m3 .* sqrt((2*l1+1)*(2*l2+1)*(2*l3+1)/(4*pi)) .* w0;
        
        % Compute w3j(l1, l2, l3, m1, m2, -m3)
        w_m = vectorized_w3j_magnetic(l1, l2, l3, m1, m2, -m3, lf);
        
        values(mask) = pref .* w_m;
    end
end

function w0 = fast_w3j_zero(l1, l2, l3, lf)
    J = l1 + l2 + l3;
    if mod(J, 2) ~= 0, w0 = 0; return; end
    g = J/2;
    % Log-space Racah for the zero-case
    log_w0 = 0.5 * (lf(2*g-2*l1+1) + lf(2*g-2*l2+1) + lf(2*g-2*l3+1) - lf(2*g+2)) + ...
             lf(g+1) - (lf(g-l1+1) + lf(g-l2+1) + lf(g-l3+1));
    w0 = (-1)^g * exp(log_w0);
end

function w = vectorized_w3j_magnetic(l1, l2, l3, m1, m2, m3, lf)
    num_m = numel(m1);
    w = zeros(num_m, 1);
    
    % 1. The Delta Coefficient (Triangle Coefficient)
    % This is the constant part of the 3j symbol for a fixed l-triplet
    log_delta = 0.5 * (lf(l1+l2-l3+1) + lf(l1-l2+l3+1) + lf(-l1+l2+l3+1) - lf(l1+l2+l3+2));
    
    for k = 1:num_m
        % Basic check for magnetic sum
        if abs(m1(k)+m2(k)+m3(k)) > 1e-14, continue; end
        
        % 2. Limits for t (The Racah Sum)
        t_min = max([0, l2-l3-m1(k), l1-l3+m2(k)]);
        t_max = min([l1-m1(k), l2+m2(k), l1+l2-l3]);
        
        t = t_min:t_max;
        
        % 3. The Log-Numerator for the magnetic part
        log_mag = 0.5 * (lf(l1+m1(k)+1) + lf(l1-m1(k)+1) + ...
                         lf(l2+m2(k)+1) + lf(l2-m2(k)+1) + ...
                         lf(l3+m3(k)+1) + lf(l3-m3(k)+1));
        
        % 4. The Log-Denominator for each t
        term_logs = lf(t+1) + lf(l1+l2-l3-t+1) + lf(l1-m1(k)-t+1) + ...
                    lf(l2+m2(k)-t+1) + lf(l3-l2+m1(k)+t+1) + lf(l3-l1-m2(k)+t+1);
        
        % 5. The Sign and Sum
        % Standard Wigner 3j phase: (-1)^(l1 - l2 - m3)
        phase_out = (-1)^(l1 - l2 - m3(k));
        w(k) = phase_out * exp(log_delta + log_mag) * sum((-1).^t .* exp(-term_logs));
    end
end