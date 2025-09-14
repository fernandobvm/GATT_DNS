!real(8) :: local_values(5,:), global_values(5,:,:)
!allocate(local_values(5, nTracked), global_values(5, nTracked, nproc))

! Inicializa
local_values = 0.0

! Preenche pontos locais
do i = 1, nTracked
    if((indTracked(i,2).ge.xstart(2)).and.(indTracked(i,2).le.xend(2)).and.(indTracked(i,3).ge.xstart(3)).and.(indTracked(i,3).le.xend(3))) then
        local_values(1, i) = U(indTracked(i,1),indTracked(i,2),indTracked(i,3))
        local_values(2, i) = V(indTracked(i,1),indTracked(i,2),indTracked(i,3))
        local_values(3, i) = W(indTracked(i,1),indTracked(i,2),indTracked(i,3))
        local_values(4, i) = R(indTracked(i,1),indTracked(i,2),indTracked(i,3))
        local_values(5, i) = E(indTracked(i,1),indTracked(i,2),indTracked(i,3))
    endif
enddo

! Comunicação eficiente
call MPI_GATHER(local_values, 5*nTracked, MPI_DOUBLE, global_values, 5*nTracked, MPI_DOUBLE, 0, MPI_COMM_WORLD, ierror)
  
! Processamento no root
if ((mod(tstep,logAll).eq.0).or.(stepsUntilSaving.eq.0)) then
    if(nrank.eq.0) then! Run only in the root process
        open(2,file='../log.txt', status='unknown', access='append')
            if (stepsUntilSaving.eq.0) then
                nSaveTemp = nSave
            else
                nSaveTemp = 0
            endif
            write(2,'(I10,A1,I10,A1,ES13.8E1,A1,ES10.5E1,A1,F8.5,5(A1,ES10.4E2))',advance='no') nSaveTemp, char(9), tStep, char(9), t, char(9), dt, char(9), cfl, char(9), maxChange(1), char(9), maxChange(2), char(9), maxChange(3), char(9), maxChange(4), char(9), maxChange(5)

            do i = 1, nTracked
                ! Encontra processo com dados não-zero
                do j = 1, nproc
                    if (any(global_values(:,i,j) /= 0.0)) then

                        trackedValues = global_values(:,i,j)

						if (trackedNorm.ne.0) then
							trackedValues(4) = trackedValues(4)-1
							trackedValues(5) = trackedValues(5)/trackedNorm-1
						endif
						
						do k = 1,5
							if (trackedValues(k).eq.0) then
								write(2,'(A1,A1)',advance='no') char(9), "0"
							else
								write(2,'(A1,ES15.8E2)',advance='no') char(9), trackedValues(k)
							endif
						enddo

                    endif
                enddo
            enddo
        close(2)
    endif
endif