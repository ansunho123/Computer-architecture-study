CC=gcc-11
CCFLAG= -O2 -static -fno-reorder-blocks-and-partition -fno-reorder-functions -fno-reorder-blocks -fno-inline-small-functions -fno-shrink-wrap -march=native -mavx512f
CCFLAG_DEBUG=-g -O0 -static -march=native -mavx512f

SRC = matrix_multiplication.c sem.c
INC = ./


cmp: 
	$(CC) $(CCFLAG) $(SRC) -I$(INC) -o matrix_multiplication 

run:
	./matrix_multiplication all

opt: 
	$(CC) $(CCFLAG) $(SRC) -I$(INC) -o matrix_multiplication
	./matrix_multiplication opt

clean:
	@rm -rf matrix_multiplication matrix_multiplication_debug

all:
	$(CC) $(CCFLAG) $(SRC) -I$(INC) -o matrix_multiplication 
	./matrix_multiplication all

debug: 
	$(CCFLAG_DEBUG) $(SRC) -I$(INC) -o matrix_multiplication_debug 
