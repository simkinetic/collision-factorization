function T_c2r = complex2real(L_max)
    % L_max: maximum degree (L)
    % T_c2r: (L_max+1)^2 x (L_max+1)^2 block-diagonal transformation matrix
    
    % Prepare cell array for each degree l
    T_blocks = cell(L_max + 1, 1);
    
    for l = 0:L_max
        dim = 2*l + 1;
        if l == 0
            T_blocks{l+1} = 1;
            continue;
        end
        
        % m values for this degree: [-l, ..., -1, 0, 1, ..., l]
        m = (1:l)'; 
        
        % Precompute the common factor
        invSqrt2 = 1/sqrt(2);
        
        % The transformation follows:
        % Y_real_neg_m =  i/sqrt(2) * (Y_comp_-m - (-1)^m * Y_comp_m)
        % Y_real_0     =  Y_comp_0
        % Y_real_pos_m =  1/sqrt(2) * (Y_comp_-m + (-1)^m * Y_comp_m)
        
        % Diagonal components
        % Order: [m_comp = -l to -1] , [0] , [m_comp = 1 to l]
        diag_vals = [1i*ones(l,1); 1; ((-1).^m)] * invSqrt2;
        diag_vals(l+1) = 1; % m=0 is exactly 1 (not 1/sqrt(2))
        
        % Anti-diagonal components (flipping across the m=0 center)
        adiag_vals = [-1i*((-1).^flipud(m)); 1; ones(l,1)] * invSqrt2;
        adiag_vals(l+1) = 0; % Central element already handled by diagonal
        
        % Build the block: T = diag(d) + rot90(diag(ad))
        T_blocks{l+1} = diag(diag_vals) + fliplr(diag(adiag_vals));
    end
    
    % Assemble all blocks into the final sparse-capable matrix
    T_c2r = blkdiag(T_blocks{:});
end