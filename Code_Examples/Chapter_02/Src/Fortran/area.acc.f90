program area_mandelbrot
  implicit none
!
  integer, parameter :: sp = kind(1.0)   
  integer, parameter :: dp = kind(1.0d0)
  integer :: i, j, iter, numoutside
  logical :: found
  integer, parameter :: npoints = 4000, maxiter = 4000
  real (kind=dp) :: area, error
  complex (kind=dp) :: c , z
  real :: starttime, endtime
!
! Calculate area of mandelbrot set
!
!    Outer loops runs over npoints, initialize z=c
!
!    Inner loop has the iteration z=z*z+c, and threshold test
!

  numoutside = 0 

  call cpu_time(starttime)
  
  !$acc parallel loop gang reduction(+:numoutside) private(c,z,iter,i,j,found)
  do i = 0,npoints-1
     !$acc loop vector
     do j= 0,npoints-1 
        c = cmplx(-2.0+(2.5*i)/npoints + 1.0d-07,(1.125*j)/npoints + 1.0d-07)
        z = c
        iter = 0
        found = .false.
        do while (iter < maxiter .and. .not. found) 
           z = z*z + c 
           iter = iter + 1
           if (real(z)*real(z)+aimag(z)*aimag(z) > 4) then
              numoutside = numoutside + 1 
              found = .true.
           endif
        end do 
     end do
  end do


  call cpu_time(endtime)
!
! Output results
!
  area = 2.0*2.5*1.125 * real(npoints*npoints-numoutside)/real(npoints*npoints)
  error = area/real(npoints)
  print *, "Area of Mandelbrot set = ",area," +/- ",error
  print *, "Time taken for calculation: ", endtime - starttime
  !
  stop
end program area_mandelbrot
