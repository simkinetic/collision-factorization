function [N_points, actual_degree] = get_required_lebedev_points(req_deg)
    % GET_REQUIRED_LEBEDEV_POINTS Determines the minimum number of Lebedev 
    % quadrature points needed to exactly integrate the Galerkin-Petrov 
    % 3D collision tensor for a given maximum Cartesian polynomial degree.
    
    % Standard Lebedev configurations: [Algebraic Degree, Number of Points]
    leb_map = [
        3,    6;
        5,   14;
        7,   26;
        9,   38;
       11,   50;
       13,   74;
       15,   86;
       17,  110;
       19,  146;
       21,  170;
       23,  194;
       25,  230;
       27,  266;
       29,  302;
       31,  350;
       35,  434;
       41,  590;
       47,  770;
       53,  974;
       59, 1202;
       65, 1454;
       71, 1730;
       77, 2030;
       83, 2354;
       89, 2702;
       95, 3074;
      101, 3470;
      107, 3890;
      113, 4334;
      119, 4802;
      125, 5294;
      131, 5810
    ];
    
    % Find the first Lebedev rule that meets or exceeds the required degree
    idx = find(leb_map(:, 1) >= req_deg, 1, 'first');
    
    if isempty(idx)
        error('Required degree %d exceeds maximum available Lebedev degree (131).', req_deg);
    end
    
    N_points = leb_map(idx, 2);
    actual_degree = leb_map(idx, 1);
end