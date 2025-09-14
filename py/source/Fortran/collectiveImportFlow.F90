! TODO: Verificar se a varredura matricial está sendo feito primeiramente por linha ou coluna e adequar ao recomendado para o FORTRAN

    ! integer, allocatable :: dimensions(:), sendcounts(:), displs(:)
    allocate(Ug(nx,ny,nz))
    allocate(Vg(nx,ny,nz))
    allocate(Wg(nx,ny,nz))
    allocate(Rg(nx,ny,nz))
    allocate(Eg(nx,ny,nz))
    allocate(dimensions(5*nproc), sendcounts(nproc), displs(nproc))

    allocate(sliceSizes(nproc-1))
    allocate(slicesJStarts(nproc-1))
    allocate(slicesJEnds(nproc-1))
    allocate(slicesKStarts(nproc-1))
    allocate(slicesKEnds(nproc-1))

    if(nrank.eq.0) then! Run only in the root process
        
        allocate(insideWall(nx,ny,nz))
        
        call readFlow(nSave,t,Ug,Vg,Wg,Rg,Eg)
		
        do i = 1,nx
            do j = 1,ny
                do k = 1,nz
                    insideWall(i,j,k) = isnan(Ug(i,j,k))
                    if (insideWall(i,j,k)) then
                        NaN = Ug(i,j,k)
                        Ug(i,j,k) = 0
                        Vg(i,j,k) = 0
                        Wg(i,j,k) = 0
                        Rg(i,j,k) = 1
                        Eg(i,j,k) = 1
                    endif
                enddo
            enddo
        enddo

    endif

    !First, exchange data about the slice size
    ! [xstart_2 xend_2 xstart_3 xend_3 size ... xstart_2 xend_2 xstart_3 xend_3 size ... xstart_2 xend_2 xstart_3 xend_3 size]
    ! integer :: domain_dims(5)
    domain_dims = [ xstart(2), xend(2), xstart(3), xend(3), xsize(1)*xsize(2)*xsize(3) ]
    call MPI_GATHER(domain_dims, 5, MPI_INT, dimensions, 5, MPI_INT, 0, MPI_COMM_WORLD, ierror)                 

    !Then, exchange flow data 
    call MPI_Bcast(t, 1, MPI_REAL8, 0, MPI_COMM_WORLD, ierror)
    call MPI_Bcast(tstep, 1, MPI_INT, 0, MPI_COMM_WORLD, ierror)

    if(nrank.eq.0) then 
        displs(1) = 0
        do i = 1, nproc
            sendcounts(i)  = dimensions(5*i)      
            if (i > 1) then
                displs(i) = displs(i-1) + sendcounts(i-1) 

                slicesJStarts(i-1) = dimensions(5*(i-1) + 1)
                slicesJEnds(i-1)   = dimensions(5*(i-1) + 2)
                slicesKStarts(i-1) = dimensions(5*(i-1) + 3)
                slicesKEnds(i-1)   = dimensions(5*(i-1) + 4)
                sliceSizes(i-1)    = dimensions(5*(i-1) + 5)
            endif
        enddo
    endif

    call MPI_SCATTERV(Ug, sendcounts, displs, MPI_REAL8, U, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, MPI_COMM_WORLD, ierror)  
    call MPI_SCATTERV(Vg, sendcounts, displs, MPI_REAL8, V, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, MPI_COMM_WORLD, ierror)  
    call MPI_SCATTERV(Wg, sendcounts, displs, MPI_REAL8, W, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, MPI_COMM_WORLD, ierror)  
    call MPI_SCATTERV(Rg, sendcounts, displs, MPI_REAL8, R, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, MPI_COMM_WORLD, ierror)  
    call MPI_SCATTERV(Eg, sendcounts, displs, MPI_REAL8, E, xsize(1)*xsize(2)*xsize(3), MPI_REAL8, 0, MPI_COMM_WORLD, ierror)  