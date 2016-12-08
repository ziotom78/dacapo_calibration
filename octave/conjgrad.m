function [x] = conjgrad(A, b, x)
    r = b - A * x;
    p = r;
    rsold = r' * r;
    for i=1:length(b)
        Ap = A * p;
        gamma = rsold / (p' * Ap);
        x = x + gamma * p;
        r = r - gamma * Ap;
        rsnew = r' * r;
        if sqrt(rsnew) < 1e-10
            break;
        end

        p = r + (rsnew / rsold) * p;
        rsold = rsnew;
    end
end
