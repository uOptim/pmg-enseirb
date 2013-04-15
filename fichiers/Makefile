
ARCH            := $(shell uname -s | tr a-z A-Z)

PROG	:=	atoms

CFLAGS	:=	-g -O3 -std=c99
ifeq ($(ARCH),DARWIN)
CFLAGS	+=	-I /opt/local/include
LDFLAGS	+=	-L /opt/local/include
LDLIBS	+=	-framework GLUT -framework OpenGL -framework OpenCL
else
LDLIBS		:= -lOpenCL -lGL -lGLU -lglut
endif

.phony: default clean

default: $(PROG)

$(PROG): main.o vbo.o atom.o ocl.o
	$(CC) -o $@ $(LDFLAGS) $^ $(LDLIBS)

main.o: atom.h vbo.h ocl.h

atom.o: atom.h

vbo.o: vbo.h atom.h ocl.h

ocl.o: ocl.h vbo.h atom.h

clean:
	rm -rf *.o $(PROG)