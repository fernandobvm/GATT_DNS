CCOMP = gcc
FCOMP = mpifort # h5pfc

# ARGS = -ffree-line-length-0
# ARGS = -ffree-line-length-0 -ffpe-trap=invalid,zero,overflow
ARGS = -ffree-line-length-0 -O3 -march=native -ftree-vectorize -funroll-loops

CFLAGS = -fPIC -O2
HDF5_INCLUDE_DIR = /usr/include/hdf5/openmpi
HDF5_LIB_DIR = /usr/lib/x86_64-linux-gnu/hdf5/openmpi

SOURCEFOLDER = ../../../source/Fortran

include makefile_extra

include $(DECOMPDIR)/src/Makefile.inc

CLIB_OBJ = cpuTime.o
CLIB_SRC = $(SOURCEFOLDER)/cpuTime.c

INCLUDE = -I$(SOURCEFOLDER) -I$(DECOMPDIR)/include -I$(HDF5_INCLUDE_DIR)
LINKS = -L$(DECOMPDIR)/lib -l2decomp_fft -L$(HDF5_LIB_DIR) -lhdf5_fortran -lhdf5_hl -lhdf5 -lcrypto -lcurl -lpthread -lsz -lz -ldl -lm
OBJECTS = readWriteHDF5.o derivs.o nseq.o timeIntegration.o

main: $(CLIB_OBJ) disturbances.o $(OBJECTS)
	$(FCOMP) $(ARGS) -g -J. $(OBJECTS) disturbances.o $(CLIB_OBJ) $(SOURCEFOLDER)/main.F90 $(LINKS) -o main $(INCLUDE)  
	
$(CLIB_OBJ): $(CLIB_SRC)
	$(CCOMP) $(CFLAGS) -c $(CLIB_SRC) -o $(CLIB_OBJ)

disturbances.o:
	$(FCOMP) $(ARGS) -g -J. -c disturbances.F90 -o disturbances.o  
	 
%.o: $(SOURCEFOLDER)/%.F90
	$(FCOMP) $(ARGS) -g -c -J. $< $(INCLUDE) -o $@

clean:
	rm -f main *.o *.mod $(CLIB_OBJ) 

# sudo apt-get update
# sudo apt-get install libhdf5-openmpi-dev libhdf5-mpich-dev