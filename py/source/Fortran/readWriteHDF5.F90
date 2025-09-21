module readWriteHDF5
  use hdf5
  use decomp_2d
  use mpi
  implicit none
  
  private
  public :: readFlow, writeFlow, readSFD, &
            readMeanFlow, writeMeanFlow, setup_hdf5_io, h5close
  
  integer :: i,j,k
  integer(HID_T) :: plist_id
  logical :: hdf5_initialized = .false.
  integer(HSIZE_T), dimension(3) :: dims_global, dims_local, offset
  
contains

  subroutine setup_hdf5_io()
    integer :: error

    if (.not. hdf5_initialized) then
      call h5open_f(error)
      call h5pcreate_f(H5P_FILE_ACCESS_F, plist_id, error)
      call h5pset_fapl_mpio_f(plist_id, MPI_COMM_WORLD, MPI_INFO_NULL, error)

      hdf5_initialized = .true.
    endif

  end subroutine setup_hdf5_io

  subroutine h5close()
    integer :: error

    if (hdf5_initialized) then
      call h5pclose_f(plist_id, error)
      call h5close_f(error)
      hdf5_initialized = .false.
    endif
  end subroutine h5close

  subroutine readSFD(sfd_X, nx, ny, nz)
    integer :: error, mpierr
    integer(HID_T) :: file_id
    integer, intent(in) :: nx, ny, nz
    character(len=100) :: filename = 'SFD_X.h5'
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(out) :: sfd_X
    
    ! Configurar dimensões para leitura paralela
    dims_global = [nx, ny, nz]
    dims_local = [xsize(1), xsize(2), xsize(3)]
    offset = [xstart(1)-1, xstart(2)-1, xstart(3)-1]
    
    ! Inicializar HDF5 paralelo
    call setup_hdf5_io()
    
    ! Abrir arquivo em paralelo
    call h5fopen_f(filename, H5F_ACC_RDONLY_F, file_id, error, access_prp=plist_id)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5fopen_f failed for ', trim(filename), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Ler campo em paralelo
    call read_3d_field(file_id, "SFD_X", sfd_X)
    
    ! Fechar arquivo
    call h5fclose_f(file_id, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5fclose_f failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
  end subroutine readSFD

  subroutine readFlow(timeStep, t, U, V, W, R, E, insideWall, nx, ny, nz, NaN)
    real*8, intent(out) :: t
    real*8, intent(inout) :: NaN
    integer, intent(in) :: timeStep
    integer, intent(in) :: nx, ny, nz
    logical, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(inout) :: insideWall
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(inout) :: U, V, W, R, E

    character(len=10) :: timeChar
    character(len=100) :: filename
    
    ! Criar filename
    write(timeChar, fmt='(i10.10)') timeStep
    filename = '../flow_'//trim(timeChar)//'.h5'

    call readFile(filename, t, U, V, W, R, E, insideWall, nx, ny, nz, NaN)

  end subroutine readFlow

  subroutine readMeanFlow(U, V, W, R, E, insideWall, nx, ny, nz, NaN)
    real*8, intent(inout) :: NaN
    integer, intent(in) :: nx, ny, nz
    logical, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(inout) :: insideWall
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(inout) :: U, V, W, R, E
    
    real*8 :: t
    character(len=100) :: filename = '../meanflowSFD.h5'
    
    call readFile(filename, t, U, V, W, R, E, insideWall, nx, ny, nz, NaN)

  end subroutine readMeanFlow

  subroutine writeFlow(timeStep, t, U, V, W, R, E, insideWall, nx, ny, nz, NaN)
    real*8, intent(out) :: t
    real*8, intent(in) :: NaN
    integer, intent(in) :: timeStep
    integer, intent(in) :: nx, ny, nz
    logical, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(in) :: insideWall
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(in) :: U, V, W, R, E
    
    character(len=100) :: filename

    ! Criar filename
    if (timeStep < 0) then
      filename = "../flow_diverged.h5"
    else
      write(filename, '("../flow_", i10.10, ".h5")') timeStep
    endif

    call writeFile(filename, t, U, V, W, R, E, insideWall, nx, ny, nz, NaN)

  end subroutine writeFlow

  subroutine writeMeanFlow(U, V, W, R, E, insideWall, nx, ny, nz, NaN)
    real*8 :: t
    real*8, intent(in) :: NaN
    integer, intent(in) :: nx, ny, nz
    logical, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(in) :: insideWall
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(in) :: U, V, W, R, E
    
    character(len=100) :: filename = '../meanflowSFD.h5'
    
    call writeFile(filename, t, U, V, W, R, E, insideWall, nx, ny, nz, NaN)
    
  end subroutine writeMeanFlow

  subroutine readFile(filename, t, U, V, W, R, E, insideWall, nx, ny, nz, NaN)
    real*8, intent(out) :: t
    real*8, intent(inout) :: NaN
    integer, intent(in) :: nx, ny, nz
    character(len=100), intent(in) :: filename
    logical, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(inout) :: insideWall
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(inout) :: U, V, W, R, E

    integer :: error, mpierr
    integer(HSIZE_T) :: dims_scalar(1) = [1]
    integer(HID_T) :: file_id, dset_id, filespace, memspace

    ! Configurar dimensões para leitura paralela
    dims_global = [nx, ny, nz]
    dims_local = [xsize(1), xsize(2), xsize(3)]
    offset = [xstart(1)-1, xstart(2)-1, xstart(3)-1]

    ! print *, 'Rank', nrank, ' readFlow offset: ', offset
    ! print *, 'Rank', nrank, ' readFlow dims_local: ', dims_local
    ! print *, 'Rank', nrank, ' readFlow [nx, ny, nz]: ', [nx, ny, nz]

    ! Inicializar HDF5 paralelo
    call setup_hdf5_io()
    
    ! Abrir arquivo em paralelo
    print *, "Opening file: ", filename, '[nx, ny, nz]: ', [nx, ny, nz]
    call h5fopen_f(trim(filename), H5F_ACC_RDONLY_F, file_id, error, access_prp=plist_id)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5fopen_f failed for ', trim(filename), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Ler tempo (apenas rank 0)
    if (nrank == 0) then
        call h5dopen_f(file_id, "t", dset_id, error)
        if (error /= 0) then
            print *, 'Rank', nrank, ': h5dopen_f failed for dataset t (error=', error, ')'
            call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
        endif
        call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, t, dims_scalar, error)
        if (error /= 0) then
            print *, 'Rank', nrank, ': h5dread_f failed for dataset t (error=', error, ')'
            call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
        endif
        call h5dclose_f(dset_id, error)
    endif

    ! Broadcast do tempo para todos os processos
    call MPI_BCAST(t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': MPI_BCAST failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Ler campos em paralelo
    call read_3d_field(file_id, "U", U)
    call read_3d_field(file_id, "V", V)
    call read_3d_field(file_id, "W", W)
    call read_3d_field(file_id, "R", R)
    call read_3d_field(file_id, "E", E)
    
    ! Processando insideWall
    do k = xstart(3), xend(3)
        do j = xstart(2), xend(2)
            do i = xstart(1), xend(1)
                insideWall(i,j,k) = isnan(U(i,j,k))
                if (insideWall(i,j,k)) then
                    NaN = U(i,j,k)
                    U(i,j,k) = 0
                    V(i,j,k) = 0
                    W(i,j,k) = 0
                    R(i,j,k) = 1
                    E(i,j,k) = 1
                endif
            enddo
        enddo
    enddo

    ! Fechar arquivo
    call h5fclose_f(file_id, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5fclose_f failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
  end subroutine readFile

  subroutine writeFile(filename, t, U, V, W, R, E, insideWall, nx, ny, nz, NaN)
    real*8, intent(in) :: t
    real*8, intent(in) :: NaN
    integer, intent(in) :: nx, ny, nz
    logical, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(in) :: insideWall
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(in) :: U, V, W, R, E
    
    integer :: error, mpierr
    integer(HSIZE_T) :: dims_scalar(1) = [1]
    character(len=100), intent(in) :: filename
    integer(HID_T) :: file_id, scalar_space, scalar_dset
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)) :: Uw, Vw, Ww, Rw, Ew
    
    ! Substituir paredes por NaN localmente
    Uw = U; Vw = V; Ww = W; Rw = R; Ew = E
    do k = xstart(3), xend(3)
        do j = xstart(2), xend(2)
            do i = xstart(1), xend(1)
                if (insideWall(i,j,k)) then
                    Uw(i,j,k) = NaN
                    Vw(i,j,k) = NaN
                    Ww(i,j,k) = NaN
                    Rw(i,j,k) = NaN
                    Ew(i,j,k) = NaN
                endif
            enddo
        enddo
    enddo

    ! Configurar dimensões para escrita paralela
    dims_global = [nx, ny, nz]
    dims_local = [xsize(1), xsize(2), xsize(3)]
    offset = [xstart(1)-1, xstart(2)-1, xstart(3)-1]
    
    ! print *, 'Rank', nrank, ' writeFlow offset: ', offset
    ! print *, 'Rank', nrank, ' writeFlow dims_local: ', dims_local
    ! print *, 'Rank', nrank, ' writeFlow [nx, ny, nz]: ', [nx, ny, nz]

    ! Inicializar HDF5 paralelo
    call setup_hdf5_io()
    
    ! Criar arquivo com acesso paralelo
    call h5fcreate_f(trim(filename), H5F_ACC_TRUNC_F, file_id, error, access_prp=plist_id)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5fcreate_f failed for ', trim(filename), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Escrever tempo (apenas rank 0)
    call h5screate_simple_f(1, dims_scalar, scalar_space, error)
    if (error /= 0) then
    print *, 'Rank', nrank, ': h5screate_simple_f failed for scalar (error=', error, ')'
    call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    call h5dcreate_f(file_id, "t", H5T_NATIVE_DOUBLE, scalar_space, scalar_dset, error)
    if (error /= 0) then
    print *, 'Rank', nrank, ': h5dcreate_f failed for dataset t (error=', error, ')'
    call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    if (nrank == 0) then
    call h5dwrite_f(scalar_dset, H5T_NATIVE_DOUBLE, t, dims_scalar, error)
    if (error /= 0) then
        print *, 'Rank', nrank, ': h5dwrite_f failed for dataset t (error=', error, ')'
        call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    endif
    call h5dclose_f(scalar_dset, error)
    call h5sclose_f(scalar_space, error)

    ! Escrever campos em paralelo
    call write_3d_field(file_id, "U", U)
    call write_3d_field(file_id, "V", V)
    call write_3d_field(file_id, "W", W)
    call write_3d_field(file_id, "R", R)
    call write_3d_field(file_id, "E", E)
    
    ! Fechar arquivo
    call h5fclose_f(file_id, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5fclose_f failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
  end subroutine writeFile

  subroutine read_3d_field(file_id, field_name, field_data)
    integer(HID_T), intent(in) :: file_id
    character(len=*), intent(in) :: field_name
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(out) :: field_data
    
    integer(HID_T) :: dset_id, filespace, memspace, xfer_plist
    integer :: error, mpierr
    
    ! Abrir dataset
    call h5dopen_f(file_id, field_name, dset_id, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5dopen_f failed for ', trim(field_name), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Selecionar hyperslab no arquivo
    call h5dget_space_f(dset_id, filespace, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5dget_space_f failed for ', trim(field_name), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset, dims_local, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5sselect_hyperslab_f failed for ', trim(field_name), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Criar dataspace para memória
    call h5screate_simple_f(3, dims_local, memspace, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5screate_simple_f failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Configurar transferência paralela (usar plist local)
    call h5pcreate_f(H5P_DATASET_XFER_F, xfer_plist, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5pcreate_f (DXPL) failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    call h5pset_dxpl_mpio_f(xfer_plist, H5FD_MPIO_COLLECTIVE_F, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5pset_dxpl_mpio_f failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Ler dados
    call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, field_data, dims_local, error, &
                   mem_space_id=memspace, file_space_id=filespace, xfer_prp=xfer_plist)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5dread_f failed for ', trim(field_name), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Fechar recursos
    call h5pclose_f(xfer_plist, error)
    call h5sclose_f(memspace, error)
    call h5sclose_f(filespace, error)
    call h5dclose_f(dset_id, error)
  end subroutine read_3d_field

  subroutine write_3d_field(file_id, field_name, field_data)
    integer(HID_T), intent(in) :: file_id
    character(len=*), intent(in) :: field_name
    real*8, dimension(xstart(1):xend(1), xstart(2):xend(2), xstart(3):xend(3)), intent(in) :: field_data
    
    integer(HID_T) :: dset_id, filespace, memspace, dcpl_id, xfer_plist
    integer(HSIZE_T) :: chunk_dims(3)
    integer :: error, mpierr
    
    integer(8) :: chunk_int_local(3), chunk_int_global(3)

    ! Criar dataspace global
    call h5screate_simple_f(3, dims_global, filespace, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5screate_simple_f failed for global space (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Configurar chunking e compressão
    call h5pcreate_f(H5P_DATASET_CREATE_F, dcpl_id, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5pcreate_f (DCPL) failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif

    chunk_dims = dims_local
    call h5pset_chunk_f(dcpl_id, 3, chunk_dims, error)
    call h5pset_deflate_f(dcpl_id, 6, error)

    ! compute consistent chunk_dims across ranks
    ! integer(HSIZE_T) :: chunk_dims(3)
    ! integer(8) :: chunk_int_local(3), chunk_int_global(3)
    ! chunk_int_local = [int(dims_local(1),8), int(dims_local(2),8), int(dims_local(3),8)]
    ! call MPI_ALLREDUCE(chunk_int_local, chunk_int_global, 3, MPI_INTEGER8, MPI_MAX, MPI_COMM_WORLD, mpierr)
    ! chunk_dims = [chunk_int_global(1), chunk_int_global(2), chunk_int_global(3)]
    ! call h5pset_chunk_f(dcpl_id, 3, chunk_dims, error)
    ! ! optionally remove/comment the next line to test without compression
    ! call h5pset_deflate_f(dcpl_id, 6, error)    

    ! Criar dataset
    call h5dcreate_f(file_id, field_name, H5T_NATIVE_DOUBLE, filespace, dset_id, error, dcpl_id)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5dcreate_f failed for ', trim(field_name), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    call h5pclose_f(dcpl_id, error)
    call h5sclose_f(filespace, error)
    
    ! Selecionar hyperslab no arquivo
    call h5dget_space_f(dset_id, filespace, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5dget_space_f failed for ', trim(field_name), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    call h5sselect_hyperslab_f(filespace, H5S_SELECT_SET_F, offset, dims_local, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5sselect_hyperslab_f failed for ', trim(field_name), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Criar dataspace para memória
    call h5screate_simple_f(3, dims_local, memspace, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5screate_simple_f failed for memspace (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Configurar transferência paralela (usar plist local)
    call h5pcreate_f(H5P_DATASET_XFER_F, xfer_plist, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5pcreate_f (DXPL) failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    call h5pset_dxpl_mpio_f(xfer_plist, H5FD_MPIO_COLLECTIVE_F, error)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5pset_dxpl_mpio_f failed (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Escrever dados    
    call h5dwrite_f(dset_id, H5T_NATIVE_DOUBLE, field_data, dims_local, error, &
                    mem_space_id=memspace, file_space_id=filespace, xfer_prp=xfer_plist)
    if (error /= 0) then
      print *, 'Rank', nrank, ': h5dwrite_f failed for ', trim(field_name), ' (error=', error, ')'
      call MPI_ABORT(MPI_COMM_WORLD, error, mpierr)
    endif
    
    ! Fechar recursos
    call h5pclose_f(xfer_plist, error)
    call h5sclose_f(memspace, error)
    call h5sclose_f(filespace, error)
    call h5dclose_f(dset_id, error)
  end subroutine write_3d_field

end module readWriteHDF5