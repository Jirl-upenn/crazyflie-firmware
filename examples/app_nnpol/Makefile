# app_nnpol — onboard neural-network policy (see README.md).
#
# Standard out-of-tree app build: the firmware's main Makefile does the
# work; app-config supplies the extra Kconfig options (app layer on, OOT
# controller hook, larger app stack for the network's activations).
#
# Build (from this directory, with the repo's pixi env):
#     pixi run --manifest-path ../../pixi.toml make -j
CRAZYFLIE_BASE := ../..

OOT_CONFIG := $(PWD)/app-config

include $(CRAZYFLIE_BASE)/tools/make/oot.mk
