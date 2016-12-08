# If this map had a nonzero mean, part of the offset would have been
# incorporated by the baselines
skymap = [-0.4; 0.2; 0.2]; # This is orthogonal to D and has zero mean
P = [
  1 0 0;
  1 0 0;
  0 1 0;
  1 0 0;
  0 1 0;
  0 0 1;
  0 0 1;
  0 0 1;
  1 0 0;
  0 1 0;
  1 0 0];
G = [4.1 4.2];
Gext = [G(1), G(1), G(1), G(1), G(1), G(1), G(2), G(2), G(2), G(2), G(2)]';
b = [10.0 10.5];
D = sin(2 * pi * [0; 1.0/3.0; 2.0/3.0]);

tod = (Gext .* (P * (D + skymap))) + ...
    [b(1); b(1); b(1); b(1); b(1); b(1); b(2); b(2); b(2); b(2); ...
     b(2)]
#  ... + 0.01 * randn(11, 1)

# Start by supposing that the sky is just the dipole D
mc = [D, ones(3, 1)];

# Now throw away skymap, G and b, and hunt for their true value using the DaCapo algorithm
skymap = zeros(3, 1);
G = ones(1, 2);
G = [4.60127588, 3.78430781];
b = zeros(1, 2);

F = [1, 0, D(1), 0;
     1, 0, D(1), 0;
     1, 0, D(2), 0;
     1, 0, D(1), 0;
     1, 0, D(2), 0;
     1, 0, D(3), 0;
     0, 1, 0, D(3);
     0, 1, 0, D(3);
     0, 1, 0, D(1);
     0, 1, 0, D(2);
     0, 1, 0, D(1)];

pcond = inv(F' * F)

for iter = 0:4
    printf("DaCapo iteration: %d\n", iter)

    F = [1, 0, D(1)+skymap(1), 0;
         1, 0, D(1)+skymap(1), 0;
         1, 0, D(2)+skymap(2), 0;
         1, 0, D(1)+skymap(1), 0;
         1, 0, D(2)+skymap(2), 0;
         1, 0, D(3)+skymap(3), 0;
         0, 1, 0, D(3)+skymap(3);
         0, 1, 0, D(3)+skymap(3);
         0, 1, 0, D(1)+skymap(1);
         0, 1, 0, D(2)+skymap(2);
         0, 1, 0, D(1)+skymap(1)];

    Ptilde = [
      G(1) 0 0;
      G(1) 0 0;
      0 G(1) 0;
      G(1) 0 0;
      0 G(1) 0;
      0 0 G(1);
      0 0 G(2);
      0 0 G(2);
      G(2) 0 0;
      0 G(2) 0;
      G(2) 0 0];

    M = Ptilde' * Ptilde;
    invM = inv(M);
    mat2x2 = mc' * invM * mc;
    MCminv = invM - invM * mc * inv(mat2x2) * mc' * invM;
    Z = eye(11) - Ptilde * MCminv * Ptilde';
    A = F' * Z * F;

    # Gains and baselines
    # ahat = inv(A) * F' * Z * tod

    ahat = pconjgrad(pcond, A, F' * Z * tod, [b G]')

    b = ahat(1:2)';
    G = ahat(3:4)';

    # Map
    skymap_corr = (MCminv * Ptilde' * (tod - F * ahat));
    skymap += skymap_corr;
endfor

printf("Solution: ")
printf("%.5g  ", ahat)
