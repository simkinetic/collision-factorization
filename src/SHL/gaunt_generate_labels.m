function NZ_Indices = gaunt_generate_labels(L_max)
    % Refined Generator: Filters by Triangle, Parity, and Magnetic rules
    est_size = ceil((L_max + 1)^3 * 0.75); 
    NZ_Indices = zeros(est_size, 3, 'uint32');
    count = 0;
    get_q = @(l, m) l*l + l + m + 1;

    for l1 = 0:L_max
        for l2 = 0:L_max
            l3_min = abs(l1 - l2);
            l3_max = min(L_max, l1 + l2);
            
            % STEP BY 2: This is the critical change.
            % It ensures l1 + l2 + l3 is ALWAYS even.
            for l3 = l3_start_consistent_with_parity(l1, l2, l3_min) : 2 : l3_max
                for m1 = -l1:l1
                    q1 = get_q(l1, m1);
                    for m2 = -l2:l2
                        q2 = get_q(l2, m2);
                        
                        m3 = m1 + m2;
                        if abs(m3) <= l3
                            q3 = get_q(l3, m3);
                            count = count + 1;
                            NZ_Indices(count, :) = [q1, q2, q3];
                        end
                    end
                end
            end
        end
    end
    NZ_Indices = NZ_Indices(1:count, :);
end

function start = l3_start_consistent_with_parity(l1, l2, l3_min)
    % Ensures that (l1 + l2 + start) is even.
    if mod(l1 + l2 + l3_min, 2) == 0
        start = l3_min;
    else
        start = l3_min + 1;
    end
end