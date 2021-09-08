TOP = $(abspath ../..)
include $(TOP)/defs.mak

APPBUILDER    = $(TOP)/scripts/appbuilder
APP_NAME      = python3

ifndef APP_MODE
APP_MODE      = package
endif

APP_PATH      = /home/user/miniconda/bin/python3
PACKAGE_PATH  = myst/bin/$(APP_NAME)
ROOTFS        = ext2rootfs
TIMEOUT = 180

QUERY_INTERVAL = 2
SHUTDOWN_WAIT = 2 

ifdef STRACE
OPTS = --strace
endif

all: appdir build

appdir:
	$(APPBUILDER) -v -d Dockerfile.rootfs

###########################################
## Build
###########################################

build:
	@echo "\n-------ext2 build start--------\n"
	$(MAKE) build-ext2
	@echo "\n-------package build start--------\n"
	$(MAKE) build-package
	@echo "\n-------build end--------\n"

build-ext2: appdir
	$(MYST) mkext2 appdir $(ROOTFS)
	$(MYST) fssig --roothash $(ROOTFS) > roothash

build-package: package.pem
	$(MYST) package-sgx --roothash=roothash package.pem config.json

###########################################
## Run
###########################################

# running both service and client in one recipe
run:
	./kill.sh $(PACKAGE_PATH) $(APP_PATH)
	@echo "\n------starting Pytorch Inference Service($(APP_MODE) mode)------\n"
	$(MYST_EXEC) $(OPTS) $(ROOTFS) $(APP_PATH) --app-config-path config.json

###########################################
## Debug
###########################################

gdb:
# ifeq ($(APP_MODE),package)
# 	$(MYST_GDB) --args $(PACKAGE_PATH) --rootfs=$(ROOTFS)
# else
	$(MYST_GDB) --args $(MYST_EXEC) $(OPTS) $(ROOTFS) $(APP_PATH) --app-config-path config.json
# endif

###########################################
## Misc
###########################################

package.pem:
	openssl genrsa -out package.pem -3 3072

clean:
	rm -rf appdir myst package.pem $(ROOTFS)

test:
	$(MYST_EXEC) $(OPTS) $(ROOTFS) $(APP_PATH) --app-config-path config.json

test1:
	$(PACKAGE_PATH) $(OPTS) --rootfs=$(ROOTFS)