program serial
      implicit none

      integer, parameter             :: width=1000
      integer, parameter             :: height=1000
      double precision, parameter    :: temp_tolerance=0.01

      integer                        :: i, j, iteration=1
      double precision               :: worst_dt=100.0
      real                           :: start_time, stop_time

      double precision, dimension(0:height+1,0:width+1) :: temperature, temperature_previous

      call cpu_time(start_time)

      call initialize(temperature_previous)

      do while ( worst_dt > temp_tolerance )

         !$omp parallel do
         do j=1,width
            do i=1,height
               temperature(i,j)=0.25*(temperature_previous(i+1,j)+temperature_previous(i-1,j)+ &
                                      temperature_previous(i,j+1)+temperature_previous(i,j-1) )
            enddo
         enddo
         !$omp end parallel do

         worst_dt=0.0

         !$omp parallel do reduction(max:worst_dt)
         do j=1,width
            do i=1,height
               worst_dt = max( abs(temperature(i,j) - temperature_previous(i,j)), worst_dt )
               temperature_previous(i,j) = temperature(i,j)
            enddo
         enddo
         !$omp end parallel do

         if( mod(iteration,100).eq.0 ) then
            call track_progress(temperature, iteration)
         endif

         iteration = iteration+1

      enddo

      call cpu_time(stop_time)

      print*, 'Max error at iteration ', iteration-1, ' was ',worst_dt
      print*, 'Total time was ',stop_time-start_time, ' seconds.'

end program serial


subroutine initialize( temperature_previous )
      implicit none

      integer, parameter             :: width=1000
      integer, parameter             :: height=1000
      integer                        :: i,j

      double precision, dimension(0:height+1,0:width+1) :: temperature_previous

      temperature_previous = 0.0

      do i=0,height+1
         temperature_previous(i,0) = 0.0
         temperature_previous(i,width+1) = (100.0/height) * i
      enddo

      do j=0,width+1
         temperature_previous(0,j) = 0.0
         temperature_previous(height+1,j) = ((100.0)/width) * j
      enddo

end subroutine initialize


subroutine track_progress(temperature, iteration)
      implicit none

      integer, parameter             :: width=1000
      integer, parameter             :: height=1000
      integer                        :: i,iteration

      double precision, dimension(0:height+1,0:width+1) :: temperature

      print *, '---------- Iteration number: ', iteration, ' ---------------'
      do i=5,0,-1
         write (*,'("("i4,",",i4,"):",f6.2,"  ")',advance='no') &
                   height-i,width-i,temperature(height-i,width-i)
      enddo
      print *
end subroutine track_progress
