! Declarações adicionais
! integer :: tp_packed_size, tmp_pack_size, tp_position, tp_total_size
! integer, allocatable :: tp_sendcounts(:), tp_displs(:)
! character(len=:), allocatable :: tp_sendbuf, tp_recvbuf

t2 = MPI_Wtime()

! Calcular tamanho do buffer necessário
tp_packed_size = 0
do i = 1, nTracked
    ! Tamanho necessário: 1 inteiro (índice) + 5 valores reais
    if ((indTracked(i,2).ge.xstart(2)).and.(indTracked(i,2).le.xend(2)).and. &
        (indTracked(i,3).ge.xstart(3)).and.(indTracked(i,3).le.xend(3))) then
        
        call MPI_PACK_SIZE(1, MPI_INTEGER, MPI_COMM_WORLD, tmp_pack_size, ierror)
        tp_packed_size = tp_packed_size + tmp_pack_size

        call MPI_PACK_SIZE(5, MPI_DOUBLE, MPI_COMM_WORLD, tmp_pack_size, ierror)
        tp_packed_size = tp_packed_size + tmp_pack_size

    endif
enddo

! Alocar buffer de envio
allocate(character(len=tp_packed_size) :: tp_sendbuf)

! Empacotar dados
tp_position = 0
do i = 1, nTracked
    if ((indTracked(i,2).ge.xstart(2)).and.(indTracked(i,2).le.xend(2)).and. &
        (indTracked(i,3).ge.xstart(3)).and.(indTracked(i,3).le.xend(3))) then
        
        ! Empacotar índice do ponto
        call MPI_PACK(i, 1, MPI_INTEGER, tp_sendbuf, tp_packed_size, tp_position, MPI_COMM_WORLD, ierror)
        
        ! Empacotar valores
        trackedValues = [U(indTracked(i,1),indTracked(i,2),indTracked(i,3)), &
                         V(indTracked(i,1),indTracked(i,2),indTracked(i,3)), &
                         W(indTracked(i,1),indTracked(i,2),indTracked(i,3)), &
                         R(indTracked(i,1),indTracked(i,2),indTracked(i,3)), &
                         E(indTracked(i,1),indTracked(i,2),indTracked(i,3))]
        
        call MPI_PACK(trackedValues, 5, MPI_DOUBLE, tp_sendbuf, tp_packed_size, tp_position, MPI_COMM_WORLD, ierror)
    endif
enddo

! Coletar tamanhos de todos os buffers
allocate(tp_displs(0:nproc-1))
allocate(tp_sendcounts(0:nproc-1))

call MPI_GATHER(tp_position, 1, MPI_INTEGER, &
                tp_sendcounts, 1, MPI_INTEGER, &
                0, MPI_COMM_WORLD, ierror)

! Calcular deslocamentos no root
tp_total_size = 0
if (nrank.eq.0) then
    tp_displs(0) = 0
    do i = 1, nproc-1
        tp_displs(i) = tp_displs(i-1) + tp_sendcounts(i-1)
    enddo
    tp_total_size = tp_displs(nproc-1) + tp_sendcounts(nproc-1)
    allocate(character(len=tp_total_size) :: tp_recvbuf)
else
    allocate(character(len=1) :: tp_recvbuf)
endif

! Coletar dados compactados
call MPI_GATHERV(tp_sendbuf, tp_position, MPI_PACKED, &
                 tp_recvbuf, tp_sendcounts, tp_displs, MPI_PACKED, &
                 0, MPI_COMM_WORLD, ierror)

comm_time = comm_time + (MPI_Wtime() - t2)

! Processar no root
if (nrank.eq.0) then
    open(2, file='../log.txt', status='unknown', access='append')

    if (stepsUntilSaving.eq.0) then
        nSaveTemp = nSave
    else
        nSaveTemp = 0
    endif

    write(2, '(I10,A1,I10,A1,ES13.8E1,A1,ES10.5E1,A1,F8.5,5(A1,ES10.4E2))', advance='no') & 
        nSaveTemp, char(9), tStep, char(9), t, char(9), dt, char(9), cfl, char(9), maxChange(1), & 
        char(9), maxChange(2), char(9), maxChange(3), char(9), maxChange(4), char(9), maxChange(5)

    ! Processar dados recebidos
    tp_position = 0
    do while (tp_position < tp_total_size)
        t2 = MPI_Wtime()
        ! Desempacotar índice
        call MPI_UNPACK(tp_recvbuf, tp_total_size, tp_position, i, 1, MPI_INTEGER, MPI_COMM_WORLD, ierror)
        
        ! Desempacotar valores
        call MPI_UNPACK(tp_recvbuf, tp_total_size, tp_position, trackedValues, 5, MPI_DOUBLE, MPI_COMM_WORLD, ierror)
        
        ! Processar valores (código existente)
        if (trackedNorm /= 0) then
            trackedValues(4) = trackedValues(4) - 1
            trackedValues(5) = trackedValues(5) / trackedNorm - 1
        endif
        comm_time = comm_time + (MPI_Wtime() - t2)

        t2 = MPI_Wtime()
        do j = 1, 5
            if (trackedValues(j) == 0) then
                write(2, '(A1,A1)', advance='no') char(9), "0"
            else
                write(2, '(A1,ES15.8E2)', advance='no') char(9), trackedValues(j)
            endif
        enddo
        io_time = io_time + (MPI_Wtime() - t2)
    enddo
    
    close(2)
endif

! Limpar memória
deallocate(tp_sendbuf)
deallocate(tp_recvbuf, tp_sendcounts, tp_displs)