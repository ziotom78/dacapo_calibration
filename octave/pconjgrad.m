function [x] = pconjgrad(Minv, A, b, x)
    r = b - A * x;
    z = Minv * r;
    p = z;
    rsold = z' * r;
    iter = 0;
    for i=1:length(b)
        Ap = A * p;
        gamma = rsold / (p' * Ap);
        x = x + gamma * p;
        r = r - gamma * Ap;
        if sqrt(r' * r) < 1e-10
            break;
        end

        z = Minv * r;
        rsnew = z' * r;
        p = z + (rsnew / rsold) * p;
        rsold = rsnew;
        iter = iter + 1;
    end
end
