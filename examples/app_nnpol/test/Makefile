# The app's pure-C core for the host: libnnpol_core.so (test/nnpol_host.c
# over the same sources the firmware compiles). Used by run_host_parity.py
# and, at launch, by the ground station's parity gate.
#
#     make -C examples/app_nnpol/test          # builds build/libnnpol_core.so
#     make -C examples/app_nnpol/test check    # and runs the host parity suite

CC ?= gcc
PYTHON ?= python3
SRC := ../src
OUT ?= build
CORE := $(SRC)/nnpol_slot_format.c $(SRC)/nnpol_policy.c $(SRC)/nnpol_obs_race.c \
        $(SRC)/nnpol_action.c $(SRC)/nnpol_core.c

$(OUT)/libnnpol_core.so: nnpol_host.c $(CORE) $(SRC)/nnpol.h $(SRC)/nnpol_slot.h
	mkdir -p $(OUT)
	$(CC) -O2 -Wall -Wextra -Werror -fPIC -shared -I$(SRC) -o $@ nnpol_host.c $(CORE) -lm

check: $(OUT)/libnnpol_core.so
	$(PYTHON) run_host_parity.py

clean:
	rm -rf $(OUT)

.PHONY: check clean
