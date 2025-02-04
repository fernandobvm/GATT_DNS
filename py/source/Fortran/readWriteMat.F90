module readWriteMat

  use hdf5

  contains

	subroutine readFlow(timeStep, t, U, V, W, R, E)
		! DECLARE VARIABLES
		implicit none

		! Inputs
		integer, intent(in) :: timeStep
		! Outputs
		real*8, intent(out) :: t
		real*8, dimension(:,:,:), allocatable, intent(out) :: U, V, W, R, E
		
		! HDF5 variables
		integer(hid_t) :: file_id       ! File identifier
		integer(hid_t) :: dset_id       ! Dataset identifier
		integer(hid_t) :: dataspace     ! Dataspace identifier
		integer(hsize_t), dimension(3) :: dims ! Dataset dimensions
		integer(hsize_t), dimension(1) :: scalar_dims  ! Dimension for scalar read
		integer :: error                ! Error flag (use integer instead of hid_t)
		integer :: rank                 ! Dataset rank
		character(len=10) :: timeChar   ! Time step string
		character(len=100) :: filename  ! File name
		
		! Initialize HDF5 interface
		call h5open_f(error)
		if (error /= 0) then
			print *, "Error initializing HDF5"
			return
		endif
		
		! Create filename
		write(timeChar,fmt='(i10.10)') timeStep
		filename = '../flow_'//trim(timeChar)//'.h5'
		
		! Open HDF5 file
		call h5fopen_f(trim(filename), H5F_ACC_RDONLY_F, file_id, error)
		if (error /= 0) then
			print *, "Error opening file: ", trim(filename)
			return
		endif
		
		! Read time value (t)
		scalar_dims = (/1/)  ! Set dimension for scalar read
		call h5dopen_f(file_id, "t", dset_id, error)
		if (error /= 0) then
			print *, "Error opening dataset t"
			return
		endif
		
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, t, scalar_dims, error)
		if (error /= 0) then
			print *, "Error reading dataset t"
			return
		endif
		
		call h5dclose_f(dset_id, error)
		
		! Read dimensions from U dataset (assumed all variables have same dimensions)
		call h5dopen_f(file_id, "U", dset_id, error)
		if (error /= 0) then
			print *, "Error opening dataset U"
			return
		endif
		
		call h5dget_space_f(dset_id, dataspace, error)
		call h5sget_simple_extent_ndims_f(dataspace, rank, error)
		call h5sget_simple_extent_dims_f(dataspace, dims, dims, error)
		call h5dclose_f(dset_id, error)
		
		! If rank is 2, set third dimension to 1
		if (rank == 2) then
			dims(3) = 1
		endif
		
		! Allocate arrays
		allocate(U(dims(1),dims(2),dims(3)))
		allocate(V(dims(1),dims(2),dims(3)))
		allocate(W(dims(1),dims(2),dims(3)))
		allocate(R(dims(1),dims(2),dims(3)))
		allocate(E(dims(1),dims(2),dims(3)))
		
		
		! Read datasets
		call h5dopen_f(file_id, "U", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, U, dims, error)
		call h5dclose_f(dset_id, error)
		
		call h5dopen_f(file_id, "V", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, V, dims, error)
		call h5dclose_f(dset_id, error)
		
		call h5dopen_f(file_id, "W", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, W, dims, error)
		call h5dclose_f(dset_id, error)
		
		call h5dopen_f(file_id, "R", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, R, dims, error)
		call h5dclose_f(dset_id, error)
		
		call h5dopen_f(file_id, "E", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, E, dims, error)
		call h5dclose_f(dset_id, error)
		
		! Close HDF5 file
		call h5fclose_f(file_id, error)
		
		! Close HDF5 interface
		call h5close_f(error)
	end subroutine

	subroutine readSFD(sfd_X)
		! DECLARE VARIABLES
		implicit none

		! Outputs
		real*8, dimension(:,:,:), allocatable, intent(out) :: sfd_X
		
		! HDF5 variables
		integer(hid_t) :: file_id       ! File identifier
		integer(hid_t) :: dset_id       ! Dataset identifier
		integer(hid_t) :: dataspace     ! Dataspace identifier
		integer(hsize_t), dimension(3) :: dims ! Dataset dimensions
		integer :: error                ! Error flag
		integer :: rank                 ! Dataset rank
		character(len=100) :: filename  ! File name
		
		! Initialize HDF5 interface
		call h5open_f(error)
		if (error /= 0) then
			print *, "Error initializing HDF5"
			return
		endif
		
		! Open HDF5 file
		filename = 'SFD_X.h5'
		call h5fopen_f(trim(filename), H5F_ACC_RDONLY_F, file_id, error)
		if (error /= 0) then
			print *, "Error opening file: ", trim(filename)
			return
		endif
		
		! Read dimensions from dataset
		call h5dopen_f(file_id, "SFD_X", dset_id, error)
		if (error /= 0) then
			print *, "Error opening dataset SFD_X"
			return
		endif
		
		call h5dget_space_f(dset_id, dataspace, error)
		call h5sget_simple_extent_ndims_f(dataspace, rank, error)
		call h5sget_simple_extent_dims_f(dataspace, dims, dims, error)
		
		! If rank is 2, set third dimension to 1
		if (rank == 2) then
			dims(3) = 1
		endif
		
		! Allocate array
		allocate(sfd_X(dims(1),dims(2),dims(3)))
		
		! Read dataset
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, sfd_X, dims, error)
		if (error /= 0) then
			print *, "Error reading dataset SFD_X"
			return
		endif
		
		! Close dataset
		call h5dclose_f(dset_id, error)
		
		! Close HDF5 file
		call h5fclose_f(file_id, error)
		
		! Close HDF5 interface
		call h5close_f(error)
	end subroutine

	subroutine writeFlow(timeStep,t,U,V,W,R,E)
		! DECLARE VARIABLES
		implicit none

		! Inputs
		integer, intent(in) :: timeStep
		real*8, intent(in) :: t
		real*8, dimension(:,:,:), intent(in) :: U, V, W, R, E

		! Internals
		integer :: hdf_error
		integer(hid_t) :: file_id, dataspace_id, dataset_id, plist_id
		integer(hsize_t), dimension(3) :: dims, chunk_dims
		integer(hsize_t), dimension(1) :: dims1
		character(len=50) :: filename

		! Initialize HDF5 interface
		call h5open_f(hdf_error)
		if (hdf_error /= 0) then
			print *, "Error initializing HDF5"
			return
		endif

		! GET DATA SIZE
		dims = shape(U)
		dims1(1) = 1
		chunk_dims = dims / 2  ! Define um chunk menor para melhor compressão (ajustável)

		! DEFINE FILE NAME
		if (timeStep < 0) then
			filename = "../flow_diverged.h5"
		else
			write(filename, '("../flow_", i10.10, ".h5")') timeStep
		end if

		! CREATE OR OPEN HDF5 FILE
		call h5fcreate_f(trim(filename), H5F_ACC_TRUNC_F, file_id, hdf_error)

		! WRITE SCALAR VARIABLE "t"
		call h5screate_simple_f(1, dims1, dataspace_id, hdf_error)
		call h5dcreate_f(file_id, "t", H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, hdf_error)
		call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, t, dims1, hdf_error)
		call h5dclose_f(dataset_id, hdf_error)
		call h5sclose_f(dataspace_id, hdf_error)

		! CREATE PROPERTY LIST FOR COMPRESSION
		call h5pcreate_f(H5P_DATASET_CREATE_F, plist_id, hdf_error)
		call h5pset_chunk_f(plist_id, 3, chunk_dims, hdf_error)  ! Definir chunks
		call h5pset_deflate_f(plist_id, 8, hdf_error)  ! Aplicar compressão Gzip nível 8

		! WRITE 3D VARIABLES WITH COMPRESSION
		call h5screate_simple_f(3, dims, dataspace_id, hdf_error)
		call write_compressed_dataset(file_id, "U", U, dataspace_id, plist_id, hdf_error)
		call write_compressed_dataset(file_id, "V", V, dataspace_id, plist_id, hdf_error)
		call write_compressed_dataset(file_id, "W", W, dataspace_id, plist_id, hdf_error)
		call write_compressed_dataset(file_id, "R", R, dataspace_id, plist_id, hdf_error)
		call write_compressed_dataset(file_id, "E", E, dataspace_id, plist_id, hdf_error)
		call h5sclose_f(dataspace_id, hdf_error)

		! CLOSE PROPERTY LIST
		call h5pclose_f(plist_id, hdf_error)

		! CLOSE HDF5 FILE
		call h5fclose_f(file_id, hdf_error)
		call h5close_f(hdf_error)
	end subroutine
	
	subroutine write_compressed_dataset(file_id, name, data, dataspace_id, plist_id, hdf_error)
		implicit none
		integer(hid_t), intent(in) :: file_id, dataspace_id, plist_id
		integer, intent(out) :: hdf_error
		character(len=*), intent(in) :: name
		real*8, dimension(:,:,:), intent(in) :: data
		integer(hsize_t), dimension(3) :: dims
		integer(hid_t) :: dataset_id

		dims = shape(data)
		
		! Criar dataset com compressão
		call h5dcreate_f(file_id, name, H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, hdf_error, plist_id)
		call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, data, dims, hdf_error)
		call h5dclose_f(dataset_id, hdf_error)
	end subroutine

	subroutine readMeanFlow(U,V,W,R,E)
		! DECLARE VARIABLES
		implicit none

		! Arguments
			
		real*8, dimension(:,:,:), allocatable, intent(out) :: U, V, W, R, E
		
		! HDF5 variables
		integer(hid_t) :: file_id       ! File identifier
		integer(hid_t) :: dset_id       ! Dataset identifier
		integer(hid_t) :: dataspace     ! Dataspace identifier
		integer(hsize_t), dimension(3) :: dims ! Dataset dimensions
		integer :: error                ! Error flag
		integer :: rank                 ! Dataset rank
		character(len=100) :: filename  ! File name	

		! Initialize HDF5 interface
		call h5open_f(error)
		if (error /= 0) then
			print *, "Error initializing HDF5"
			return
		endif
		
		! Open HDF5 file
		filename = '../meanflowSFD.h5'
		call h5fopen_f(trim(filename), H5F_ACC_RDONLY_F, file_id, error)
		if (error /= 0) then
			print *, "Error opening file: ", trim(filename)
			return
		endif
		
		! Read dimensions from U dataset (assumed all variables have same dimensions)
		call h5dopen_f(file_id, "U", dset_id, error)
		if (error /= 0) then
			print *, "Error opening dataset U"
			return
		endif
		
		call h5dget_space_f(dset_id, dataspace, error)
		call h5sget_simple_extent_ndims_f(dataspace, rank, error)
		call h5sget_simple_extent_dims_f(dataspace, dims, dims, error)
		call h5dclose_f(dset_id, error)
		
		! If rank is 2, set third dimension to 1
		if (rank == 2) then
			dims(3) = 1
		endif
		
		! Allocate arrays
		allocate(U(dims(1),dims(2),dims(3)))
		allocate(V(dims(1),dims(2),dims(3)))
		allocate(W(dims(1),dims(2),dims(3)))
		allocate(R(dims(1),dims(2),dims(3)))
		allocate(E(dims(1),dims(2),dims(3)))
		
		! Read datasets
		call h5dopen_f(file_id, "U", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, U, dims, error)
		if (error /= 0) then
			print *, "Error reading dataset U"
			return
		endif
		call h5dclose_f(dset_id, error)
		
		call h5dopen_f(file_id, "V", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, V, dims, error)
		if (error /= 0) then
			print *, "Error reading dataset V"
			return
		endif
		call h5dclose_f(dset_id, error)
		
		call h5dopen_f(file_id, "W", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, W, dims, error)
		if (error /= 0) then
			print *, "Error reading dataset W"
			return
		endif
		call h5dclose_f(dset_id, error)
		
		call h5dopen_f(file_id, "R", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, R, dims, error)
		if (error /= 0) then
			print *, "Error reading dataset R"
			return
		endif
		call h5dclose_f(dset_id, error)
		
		call h5dopen_f(file_id, "E", dset_id, error)
		call h5dread_f(dset_id, H5T_NATIVE_DOUBLE, E, dims, error)
		if (error /= 0) then
			print *, "Error reading dataset E"
			return
		endif
		call h5dclose_f(dset_id, error)
		
		! Close HDF5 file
		call h5fclose_f(file_id, error)
		
		! Close HDF5 interface
		call h5close_f(error)
	end subroutine

	subroutine writeMeanFlow(U, V, W, R, E)

		! DECLARE VARIABLES
		implicit none

		! Inputs
		real*8, dimension(:,:,:), intent(in) :: U, V, W, R, E

		! Internals
		integer(hid_t) :: file_id, dataset_id, dataspace_id
		integer(hsize_t), dimension(3) :: dims
		integer :: error

		! Initialize HDF5 interface
		call h5open_f(error)
		if (error /= 0) then
			print *, "Error initializing HDF5"
			return
		endif
		
		! GET DATA SIZE
		dims = shape(U)

		! CREATE HDF5 FILE
		call h5fcreate_f("../meanflowSFD.h5", H5F_ACC_TRUNC_F, file_id,error)
		if (error /= 0) then
			print *, "Error creating file"
			stop
		end if

		! WRITE DATASET 'U'
		call writeDataset(file_id, "U", U, dims)

		! WRITE DATASET 'V'
		call writeDataset(file_id, "V", V, dims)

		! WRITE DATASET 'W'
		call writeDataset(file_id, "W", W, dims)

		! WRITE DATASET 'R'
		call writeDataset(file_id, "R", R, dims)

		! WRITE DATASET 'E'
		call writeDataset(file_id, "E", E, dims)

		! CLOSE HDF5 FILE
		call h5fclose_f(file_id, error)

		contains

			subroutine writeDataset(file_id, dataset_name, data, dims)
				! Subroutine to write a dataset to HDF5 file
				character(len=*), intent(in) :: dataset_name
				integer(hid_t), intent(in) :: file_id
				real*8, dimension(:,:,:), intent(in) :: data
				integer(hsize_t), dimension(3), intent(in) :: dims

				integer(hid_t) :: dataspace_id, dataset_id
				integer :: error

				! Create dataspace
				call h5screate_simple_f(3, dims, dataspace_id, error)
				if (error /= 0) then
					print *, "Error creating dataspace for: ", dataset_name
					stop
				end if

				! Create dataset
				call h5dcreate_f(file_id, dataset_name, H5T_NATIVE_DOUBLE, dataspace_id, dataset_id, error)
				if (error /= 0) then
					print *, "Error creating dataset: ", dataset_name
					stop
				end if

				! Write data
				call h5dwrite_f(dataset_id, H5T_NATIVE_DOUBLE, data, dims, error)
				if (error /= 0) then
					print *, "Error writing dataset: ", dataset_name
					stop
				end if

				! Close dataspace and dataset
				call h5sclose_f(dataspace_id, error)
				call h5dclose_f(dataset_id, error)
			end subroutine writeDataset

	end subroutine writeMeanFlow

end module
