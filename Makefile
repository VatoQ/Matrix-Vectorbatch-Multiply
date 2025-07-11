CC := gcc
CFLAGS := -Wall -g -ffast-math -msse2 -mfpmath=sse -ftree-vectorizer-verbose=5 -fopenmp -march=native -O3 -lm -std=c17 -fopt-info-vec-all=opt_report.txt
LDFLAGS := -fopenmp

TARGET := main

SRCS := main.c

OBJS := $(SRCS:.c=.o)

all: $(TARGET)

$(TARGET) : $(OBJS)
	$(CC) -o $@ $(CFLAGS) $(LDFLAGS) $^


%.o: %.c 
	$(CC) -c -o $@ $(CFLAGS) $^
clean:
	rm -rf $(TARGET) $(OBJS)
