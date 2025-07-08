CC := gcc
CFLAGS := -Wall -g -O2
OPTFLAGS := -fopt-info-vec-all=opt_report.txt
LDFLAGS := -fopenmp

TARGET := main

SRCS := main.c

OBJS := $(SRCS:.c=.o)

all: $(TARGET)


$(TARGET) : $(OBJS)
	$(CC) $(LDFLAGS) -o $(TARGET) $(OBJS)

%.o: %.c 
	$(CC) $(CFLAGS) $(OPTFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
