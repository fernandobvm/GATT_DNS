FCOMP = mpifort
ARGS = -ffree-line-length-0 
# ARGS = -ffree-line-length-0 -lhdf5_fortran -lhdf5

SOURCEFOLDER = ../../../source/Fortran

include makefile_extra

include $(DECOMPDIR)/src/Makefile.inc

# INCLUDE = -I$(SOURCEFOLDER) -I$(DECOMPDIR)/include -I/usr/include/hdf5/serial
INCLUDE = -I$(SOURCEFOLDER) -I$(DECOMPDIR)/include -I/usr/include/hdf5/serial -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5hl_fortran.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_hl.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5_fortran.a /usr/lib/x86_64-linux-gnu/hdf5/serial/libhdf5.a -lcrypto -lcurl -lpthread -lsz -lz -ldl -lm -Wl,-rpath -Wl,/usr/lib/x86_64-linux-gnu/hdf5/serial
LINKS = -L$(DECOMPDIR)/lib -l2decomp_fft -L/usr/local/hdf5/lib 
#INCLUDE = -I$(MATROOT)/extern/include -I$(SOURCEFOLDER) -I$(DECOMPDIR)/include
#LINKS = -L$(MATROOT)/bin/glnxa64 -lmat -lmx -Wl,-rpath=$(MATROOT)/bin/glnxa64/ -L$(DECOMPDIR)/lib -l2decomp_fft
OBJECTS = readWriteMat.o derivs.o nseq.o timeIntegration.o

main: disturbances.o $(OBJECTS)
	$(FCOMP) $(ARGS) -g -J. $(OBJECTS) disturbances.o $(SOURCEFOLDER)/main.F90 $(LINKS) -o main $(INCLUDE)  
	
disturbances.o:
	$(FCOMP) $(ARGS) -g -J. -c disturbances.F90 -o disturbances.o  
	 
%.o: $(SOURCEFOLDER)/%.F90
	$(FCOMP) $(ARGS) -g -c -J. $< $(INCLUDE) -o $@ 
