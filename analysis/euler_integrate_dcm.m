function [y, h, x] = euler_integrate_dcm(U, P, P_hrf, x0, h0)
    n = size(U.u,2);

    region_count = size(x0, 1); 
    
    x = zeros(region_count, n);    
    x(:,1) = x0;
    
    h = zeros(4, n, region_count);
    
    for r = 1:region_count
        h(:,1,r) = h0;
    end
    
    for i = 2:n
        x(:, i) = x(:,i-1) + single_step_neural(x(:,i-1), U.u(:, i-1), P) * U.dt;
        
        for r = 1:region_count
           h(:,i, r) = h(:,i-1, r) + single_step_hrf(h(:, i-1, r), x(r,i-1), P, P_hrf) * U.dt;
        end
        
        %h1(:,i) = h1(:,i-1) + single_step_hrf(h1(:, i-1), x(1,i-1), P, P_hrf) * U.dt;

        %h2(:,i) = h2(:,i-1) + single_step_hrf(h2(:, i-1), x(2,i-1), P, P_hrf) * U.dt;
    end
    
    y = zeros(2, n);
    
    for i = 1:n
        for r = 1:region_count
            y(r, i) = compute_bold_signal(h(:,i,r), P_hrf);
        end
    end
    
end
