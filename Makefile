CU=nvcc
CUFLAGS=-O2 -arch=sm_86 -Xcompiler=-O1 -Xcompiler=-march=native -Xcompiler=-fopenmp -Xcompiler=-ffast-math -Xcompiler=-ftree-vectorize -Xcompiler=-Wall -Xcompiler=-Wextra $(CUEXTRAFLAGS)

TARGETS=demo headless test
SOURCES=$(shell echo *.cu)
COMMON_OBJECTS=solver.o wtime.o

all: $(TARGETS)

demo: demo.o $(COMMON_OBJECTS)
	$(CU) $(CUFLAGS) $^ -o $@ $(LDFLAGS) -lGL -lGLU -lglut

headless: headless.o $(COMMON_OBJECTS)
	$(CU) $(CUFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.cu
	$(CU) $(CUFLAGS) -o $@ -c $<

test: test.o $(COMMON_OBJECTS)
	$(CU) $(CUFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGETS) *.o .depend *~

.depend: *.h *.cu
	$(CU) -MM $(SOURCES) >.depend

-include .depend

.PHONY: clean all
