program area_mandelbrot

  use omp_lib

  implicit none

  integer, parameter :: sp = kind(1.0)   
  integer, parameter :: dp = kind(1.0d0)
  integer :: i, j, iter, numoutside, numthreads
  integer, parameter :: npoints = 2000, maxiter = 2000
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

  starttime = omp_get_wtime()

  !$omp parallel do reduction(+:numoutside) private(i,j,iter,c,z) collapse(2)
  do i = 0,npoints-1
     do j= 0,npoints-1 
        c = cmplx(-2.0+(2.5*i)/npoints + 1.0d-07,(1.125*j)/npoints + 1.0d-07)
        z = c
        iter = 0
        do while (iter < maxiter) 
           z = z*z + c 
           iter = iter + 1
           if (real(z)*real(z)+aimag(z)*aimag(z) > 4) then
              numoutside = numoutside + 1 
              exit
           endif
        end do 
     end do
  end do

  endtime = omp_get_wtime()

  !$omp parallel
   numthreads = omp_get_num_threads()
  !$omp end parallel

!
! Output results
!
  area = 2.0*2.5*1.125 * real(npoints*npoints-numoutside)/real(npoints*npoints)
  error = area/real(npoints)
  print *, "Area of Mandelbrot set = ",area," +/- ",error
  print *, "Time taken for calculation: ",endtime-starttime
!
  stop
end program area_mandelbrot
