! Compile this file using the following command:
!
!    f2py -c -m ftnroutines --f90flags=-std=f2003 ftnroutines.f90

subroutine sum_subranges(array, subrange_lengths, output)
  integer(kind=8), dimension(:), intent(in) :: array
  integer(kind=8), dimension(:), intent(in) :: subrange_lengths
  integer(kind=8), dimension(size(subrange_lengths)), intent(out) :: output

  integer(kind=8) :: array_idx, subrange_idx
  integer(kind=8) :: i

  output = 0
  array_idx = 0

  do subrange_idx = 1, size(subrange_lengths)
     do i = 1, subrange_lengths(subrange_idx)
        output(subrange_idx) = output(subrange_idx) + array(array_idx + i)
     enddo
     array_idx = array_idx + subrange_lengths(subrange_idx)
  enddo

end subroutine sum_subranges
subroutine apply_f(offsets, gains, samples_per_ofsp, samples_per_gainp, &
     pix_idx, dipole_map, sky_map, output)

  real(kind=8), dimension(:), intent(in) :: offsets
  real(kind=8), dimension(:), intent(in) :: gains
  integer(kind=8), dimension(size(offsets)), intent(in) :: samples_per_ofsp
  integer(kind=8), dimension(size(gains)), intent(in) :: samples_per_gainp
  integer(kind=8), dimension(:), intent(in) :: pix_idx
  real(kind=8), dimension(:), intent(in) :: dipole_map
  real(kind=8), dimension(size(dipole_map)), intent(in) :: sky_map
  real(kind=8), dimension(size(pix_idx)), intent(out) :: output

  integer(kind=8) :: cur_ofsp_idx, cur_gainp_idx
  integer(kind=8) :: samples_in_ofsp, samples_in_gainp
  integer(kind=8) :: i

  cur_ofsp_idx = 1
  cur_gainp_idx = 1
  samples_in_ofsp = 0
  samples_in_gainp = 0

  do i = 1, size(output)

     output(i) = offsets(cur_ofsp_idx) +&
          (dipole_map(pix_idx(i) + 1) + sky_map(pix_idx(i) + 1)) * gains(cur_gainp_idx)

     samples_in_ofsp = samples_in_ofsp + 1

     if (samples_in_ofsp .ge. samples_per_ofsp(cur_ofsp_idx)) then
        cur_ofsp_idx = cur_ofsp_idx + 1
        samples_in_ofsp = 0
     endif
     samples_in_gainp = samples_in_gainp + 1

     if (samples_in_gainp .ge. samples_per_gainp(cur_gainp_idx)) then
        cur_gainp_idx = cur_gainp_idx + 1
        samples_in_gainp = 0
     endif
  enddo

end subroutine apply_f
subroutine apply_ft(vector, offsets, gains, samples_per_ofsp, samples_per_gainp, &
     pix_idx, dipole_map, sky_map, output)
  real(kind=8), dimension(:), intent(in) :: vector
  real(kind=8), dimension(:), intent(in) :: offsets
  real(kind=8), dimension(:), intent(in) :: gains
  integer(kind=8), dimension(size(offsets)), intent(in) :: samples_per_ofsp
  integer(kind=8), dimension(size(gains)), intent(in) :: samples_per_gainp
  integer(kind=8), dimension(size(vector)), intent(in) :: pix_idx
  real(kind=8), dimension(:), intent(in) :: dipole_map
  real(kind=8), dimension(size(dipole_map)), intent(in) :: sky_map
  real(kind=8), dimension(size(offsets) + size(gains)), intent(out) :: output

  integer(kind=8) :: cur_ofsp_idx, cur_gainp_idx
  integer(kind=8) :: samples_in_ofsp, samples_in_gainp
  integer(kind=8) :: i

  cur_ofsp_idx = 1
  cur_gainp_idx = 1
  samples_in_ofsp = 0
  samples_in_gainp = 0

  output = 0

  do i = 1, size(vector)
     output(cur_ofsp_idx) = output(cur_ofsp_idx) + vector(i)
     output(size(offsets) + cur_gainp_idx) = output(size(offsets) + cur_gainp_idx) + &
          vector(i) * (dipole_map(1 + pix_idx(i)) + sky_map(1 + pix_idx(i)))

     samples_in_ofsp = samples_in_ofsp + 1

     if (samples_in_ofsp .ge. samples_per_ofsp(cur_ofsp_idx)) then
        cur_ofsp_idx = cur_ofsp_idx + 1
        samples_in_ofsp = 0
     endif
     samples_in_gainp = samples_in_gainp + 1

     if (samples_in_gainp .ge. samples_per_gainp(cur_gainp_idx)) then
        cur_gainp_idx = cur_gainp_idx + 1
        samples_in_gainp = 0
     endif
  enddo

end subroutine apply_ft
subroutine compute_diagm_locally(gains, samples_per_gainp, pix_idx, output)
  real(kind=8), dimension(:), intent(in) :: gains
  integer(kind=8), dimension(size(gains)), intent(in) :: samples_per_gainp
  integer(kind=8), dimension(:), intent(in) :: pix_idx
  real(kind=8), dimension(:), intent(inout) :: output

  integer(kind=8) :: cur_gainp_idx, samples_in_gainp
  integer(kind=8) :: i

  output = 0
  cur_gainp_idx = 1
  samples_in_gainp = 0

  do i = 1, size(pix_idx)
     output(pix_idx(i) + 1) = output(pix_idx(i) + 1) + gains(cur_gainp_idx)**2
     samples_in_gainp = samples_in_gainp + 1

     if (samples_in_gainp .ge. samples_per_gainp(cur_gainp_idx)) then
        cur_gainp_idx = cur_gainp_idx + 1
        samples_in_gainp = 0
     endif
  enddo

end subroutine compute_diagm_locally
subroutine apply_ptilde(map_pixels, gains, samples_per_gainp, pix_idx, output)
  real(kind=8), dimension(:), intent(in) :: map_pixels
  real(kind=8), dimension(:), intent(in) :: gains
  integer(kind=8), dimension(size(gains)), intent(in) :: samples_per_gainp
  integer(kind=8), dimension(:), intent(in) :: pix_idx
  real(kind=8), dimension(size(pix_idx)), intent(out) :: output

  integer(kind=8) :: cur_gainp_idx, samples_in_gainp
  integer(kind=8) :: i

  output = 0
  cur_gainp_idx = 1
  samples_in_gainp = 0

  do i = 1, size(pix_idx)
     output(i) = output(i) + map_pixels(pix_idx(i) + 1) * gains(cur_gainp_idx)
     samples_in_gainp = samples_in_gainp + 1

     if (samples_in_gainp .ge. samples_per_gainp(cur_gainp_idx)) then
        cur_gainp_idx = cur_gainp_idx + 1
        samples_in_gainp = 0
     endif
  enddo

end subroutine apply_ptilde
subroutine apply_ptildet_locally(vector, gains, samples_per_gainp, pix_idx, output)
  real(kind=8), dimension(:), intent(in) :: vector
  real(kind=8), dimension(:), intent(in) :: gains
  integer(kind=8), dimension(size(gains)), intent(in) :: samples_per_gainp
  integer(kind=8), dimension(:), intent(in) :: pix_idx
  real(kind=8), dimension(:), intent(inout) :: output

  integer(kind=8) :: cur_gainp_idx, samples_in_gainp
  integer(kind=8) :: i

  output = 0
  cur_gainp_idx = 1
  samples_in_gainp = 0

  do i = 1, size(pix_idx)
     output(pix_idx(i) + 1) = output(pix_idx(i) + 1) + vector(i) * gains(cur_gainp_idx)
     samples_in_gainp = samples_in_gainp + 1

     if (samples_in_gainp .ge. samples_per_gainp(cur_gainp_idx)) then
        cur_gainp_idx = cur_gainp_idx + 1
        samples_in_gainp = 0
     endif
  enddo
end subroutine apply_ptildet_locally
subroutine clean_binned_map(inv_diagM, dipole_map, monopole_map, small_matr_prod, binned_map)
  real(kind=8), dimension(:), intent(in) :: inv_diagM
  real(kind=8), dimension(size(inv_diagM)), intent(in) :: dipole_map
  real(kind=8), dimension(size(inv_diagM)), intent(in) :: monopole_map
  real(kind=8), dimension(2), intent(in) :: small_matr_prod
  real(kind=8), dimension(size(inv_diagM)), intent(inout) :: binned_map

  binned_map = binned_map - inv_diagM * (dipole_map * small_matr_prod(1) + &
       monopole_map * small_matr_prod(2))

end subroutine clean_binned_map
