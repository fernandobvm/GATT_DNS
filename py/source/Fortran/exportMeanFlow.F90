        ! Gather global flow from slices
        if(nrank.eq.0) then! Run only in the root process
        
        Ug(xstart(1):xend(1),xstart(2):xend(2),xstart(3):xend(3)) = Umean
        Vg(xstart(1):xend(1),xstart(2):xend(2),xstart(3):xend(3)) = Vmean
        Wg(xstart(1):xend(1),xstart(2):xend(2),xstart(3):xend(3)) = Wmean
        Rg(xstart(1):xend(1),xstart(2):xend(2),xstart(3):xend(3)) = Rmean
        Eg(xstart(1):xend(1),xstart(2):xend(2),xstart(3):xend(3)) = Emean
        
        t3 = MPI_Wtime()
        do i = 1,nproc-1
            call MPI_RECV(Ug(xstart(1):xend(1),slicesJStarts(i):slicesJEnds(i),slicesKStarts(i):slicesKEnds(i)), sliceSizes(i), MPI_REAL8, i, 1,  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
            call MPI_RECV(Vg(xstart(1):xend(1),slicesJStarts(i):slicesJEnds(i),slicesKStarts(i):slicesKEnds(i)), sliceSizes(i), MPI_REAL8, i, 2,  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
            call MPI_RECV(Wg(xstart(1):xend(1),slicesJStarts(i):slicesJEnds(i),slicesKStarts(i):slicesKEnds(i)), sliceSizes(i), MPI_REAL8, i, 3,  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
            call MPI_RECV(Rg(xstart(1):xend(1),slicesJStarts(i):slicesJEnds(i),slicesKStarts(i):slicesKEnds(i)), sliceSizes(i), MPI_REAL8, i, 4,  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
            call MPI_RECV(Eg(xstart(1):xend(1),slicesJStarts(i):slicesJEnds(i),slicesKStarts(i):slicesKEnds(i)), sliceSizes(i), MPI_REAL8, i, 5,  MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierror)
        enddo
        print *, 'Rank', nrank, ' -> writeMeanFlow (comm):', (MPI_Wtime() - t3)
        
        ! Add nans inside walls
        do i = 1,nx
            do j = 1,ny
                do k = 1,nz
                    if (insideWall(i,j,k)) then
                        Ug(i,j,k) = NaN
                        Vg(i,j,k) = NaN
                        Wg(i,j,k) = NaN
                        Rg(i,j,k) = NaN
                        Eg(i,j,k) = NaN
                    endif
                enddo
            enddo
        enddo
        
        t3 = MPI_Wtime()
        ! Save the flow to file
        call writeMeanFlow(Ug,Vg,Wg,Rg,Eg)
        print *, 'Rank', nrank, ' -> writeMeanFlow (disk):', (MPI_Wtime() - t3)
        
        else ! If not the root, send data
            t3 = MPI_Wtime()
            call MPI_SEND(Umean, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, 1, MPI_COMM_WORLD, ierror)
            call MPI_SEND(Vmean, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, 2, MPI_COMM_WORLD, ierror)
            call MPI_SEND(Wmean, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, 3, MPI_COMM_WORLD, ierror)
            call MPI_SEND(Rmean, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, 4, MPI_COMM_WORLD, ierror)
            call MPI_SEND(Emean, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, 5, MPI_COMM_WORLD, ierror)
            print *, 'Rank', nrank, ' -> writeMeanFlow (comm):', (MPI_Wtime() - t3)
        endif
