        if(nrank.eq.0) then! Run only in the root process
            open(2,file='../log.txt', status='unknown', access='append')
                if (stepsUntilSaving.eq.0) then
                    nSaveTemp = nSave
                else
                    nSaveTemp = 0
                endif
                write(2,'(I10,A1,I10,A1,ES13.8E1,A1,ES10.5E1,A1,F8.5,5(A1,ES10.4E2))',advance='no') nSaveTemp, char(9), tStep, char(9), t, char(9), dt, char(9), cfl, char(9), maxChange(1), char(9), maxChange(2), char(9), maxChange(3), char(9), maxChange(4), char(9), maxChange(5)
                
                do i = 1,nTracked
                    ! If the tracked node is in the root process, gather it
                    if((indTracked(i,2).ge.xstart(2)).and.(indTracked(i,2).le.xend(2)).and.(indTracked(i,3).ge.xstart(3)).and.(indTracked(i,3).le.xend(3))) then
                        trackedValues(1) = U(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                        trackedValues(2) = V(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                        trackedValues(3) = W(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                        trackedValues(4) = R(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                        trackedValues(5) = E(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                    else ! If not, receive it
                        call MPI_RECV(trackedValues, 5, MPI_DOUBLE, MPI_ANY_SOURCE, 11+i,  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
                    endif
                    
                    if (trackedNorm.ne.0) then
                        trackedValues(4) = trackedValues(4)-1
                        trackedValues(5) = trackedValues(5)/trackedNorm-1
                    endif
                    
                    do j = 1,5
                        if (trackedValues(j).eq.0) then
                            write(2,'(A1,A1)',advance='no') char(9), "0"
                        else
                            write(2,'(A1,ES15.8E2)',advance='no') char(9), trackedValues(j)
                        endif
                    enddo
                    
                enddo
                
                write(2,*) ''
            close(2)
            
        else ! If this is not root, send data on the tracked values
            do i = 1,nTracked
                if((indTracked(i,2).ge.xstart(2)).and.(indTracked(i,2).le.xend(2)).and.(indTracked(i,3).ge.xstart(3)).and.(indTracked(i,3).le.xend(3))) then
                    trackedValues(1) = U(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                    trackedValues(2) = V(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                    trackedValues(3) = W(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                    trackedValues(4) = R(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                    trackedValues(5) = E(indTracked(i,1),indTracked(i,2),indTracked(i,3))
                    call MPI_SEND(trackedValues, 5, MPI_DOUBLE, 0, 11+i,  MPI_COMM_WORLD, ierror)
                endif
            enddo
        endif