FCOMP = mpifort # h5pfc
ARGS = -ffree-line-length-0
# ARGS = -ffree-line-length-0 -ffpe-trap=invalid,zero,overflow

HDF5_INCLUDE_DIR = /usr/include/hdf5/openmpi
HDF5_LIB_DIR = /usr/lib/x86_64-linux-gnu/hdf5/openmpi

SOURCEFOLDER = ../../../source/Fortran

include makefile_extra

include $(DECOMPDIR)/src/Makefile.inc

INCLUDE = -I$(SOURCEFOLDER) -I$(DECOMPDIR)/include -I$(HDF5_INCLUDE_DIR)
LINKS = -L$(DECOMPDIR)/lib -l2decomp_fft -L$(HDF5_LIB_DIR) -lhdf5_fortran -lhdf5_hl -lhdf5 -lcrypto -lcurl -lpthread -lsz -lz -ldl -lm
OBJECTS = readWriteHDF5.o derivs.o nseq.o timeIntegration.o

main: disturbances.o $(OBJECTS)
	$(FCOMP) $(ARGS) -g -J. $(OBJECTS) disturbances.o $(SOURCEFOLDER)/main.F90 $(LINKS) -o main $(INCLUDE)  
	
disturbances.o:
	$(FCOMP) $(ARGS) -g -J. -c disturbances.F90 -o disturbances.o  
	 
%.o: $(SOURCEFOLDER)/%.F90
	$(FCOMP) $(ARGS) -g -c -J. $< $(INCLUDE) -o $@

# sudo apt-get install libhdf5-openmpi-dev libhdf5-mpich-dev