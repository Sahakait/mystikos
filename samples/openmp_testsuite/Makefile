TOP=$(abspath ../..)
include $(TOP)/defs.mak

CONTAINER_NAME=omp.myst.apline.062021

ID=$(shell docker run -d $(CONTAINER_NAME))

ifdef STRACE
OPTS = --strace
endif

OPTS += --memory-size=256m

APPDIR=$(CURDIR)/appdir

ROOTFS=$(CURDIR)/rootfs

EXT2FS_SIZE=2147483648

TEST_FILE = passed.tests

all: $(APPDIR) $(ROOTFS)

$(ROOTFS):
	sudo $(MYST) mkext2 --force --size $(EXT2FS_SIZE) $(APPDIR) $(ROOTFS)
	sudo chown $(USER).$(USER) $(ROOTFS)
	truncate --size=-4096 $(ROOTFS)

$(APPDIR):
	myst-appbuilder Dockerfile

TESTS=$(shell cat $(TEST_FILE))

define NL


endef

tests:
	$(RUNTEST) $(MYST_EXEC) $(OPTS) $(ROOTFS) /bin/run /gcc-7.5.0/libgomp/testsuite/$(TEST_FILE)  

target19: 
	$(MYST_EXEC) $(OPTS) $(ROOTFS) /gcc-7.5.0/libgomp/testsuite/libgomp.c/target-19.c.exe

clean:
	rm -rf appdir image.tar rootfs
