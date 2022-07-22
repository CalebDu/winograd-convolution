MODE = BASE
ARCH = 86
PROJECT = Winograd

all:
	nvcc -std=c++14 main.cu -lcudnn -m64 -arch=compute_${ARCH} -code=sm_${ARCH} -o ${PROJECT}
debug:
	nvcc -g -G -std=c++14 main.cu -lcudnn -m64 -arch=compute_${ARCH} -code=sm_${ARCH} -o ${PROJECT}	
clean:
	rm ${PROJECT}
