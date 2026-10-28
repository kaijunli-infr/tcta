# Verilator harness for the TCTA core.
#   make build                 # Verilate (TOKEN_CHECK=0)
#   make run PROG=gemm         # run one vector directory and compare against the simulator's expectations
#   make all                   # every program (neg_untokened with TOKEN_CHECK=1)
#   make all FORMAL=1          # the same with the formal properties compiled as runtime assertions (J6 fallback)
#   make lint                  # verilator --lint-only on the core
VERILATOR ?= verilator
PYTHON    ?= $(HOME)/Projects/tilelang-cutoff/.venv/bin/python
SRC_DIR   := ../src
TOKEN_CHECK ?= 0
FORMAL    ?= 0          # 1: compile the `ifdef FORMAL properties (token_fifo, bank arbiter) as runtime assertions
MEMLAT    ?= 24
OBJ       := obj_dir_tc$(TOKEN_CHECK)$(if $(filter 1,$(FORMAL)),_formal,)

PKGS := $(SRC_DIR)/tcta_isa_pkg.sv $(SRC_DIR)/tcta_pkg.sv
SRCS := $(PKGS) $(wildcard $(SRC_DIR)/fp/*.sv) $(SRC_DIR)/sfu.sv $(SRC_DIR)/token_fifo.sv $(SRC_DIR)/token_net.sv \
        $(SRC_DIR)/scratchpad.sv $(SRC_DIR)/tile_meta.sv $(SRC_DIR)/token_check.sv $(SRC_DIR)/dequant.sv \
        $(SRC_DIR)/dma_load.sv $(SRC_DIR)/dma_store.sv $(SRC_DIR)/mma_pe.sv $(SRC_DIR)/mma_unit.sv $(SRC_DIR)/vec_unit.sv \
        $(SRC_DIR)/front_end.sv $(SRC_DIR)/queue.sv $(SRC_DIR)/tcta_core.sv

VFLAGS := --cc --exe --build -j 4 -O2 --top-module tb_top -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
          -Wno-UNUSEDPARAM -Wno-DECLFILENAME -Wno-CASEINCOMPLETE -Wno-UNOPTFLAT --x-assign fast --x-initial fast \
          +define+TOKEN_CHECK=$(TOKEN_CHECK) --Mdir $(OBJ) -CFLAGS "-O2" \
          $(if $(filter 1,$(FORMAL)),+define+FORMAL --assert,)

.PHONY: build run all lint clean vectors

build: $(OBJ)/Vtb_top

$(OBJ)/Vtb_top: $(SRCS) tb_top.sv tb_main.cpp
	$(VERILATOR) $(VFLAGS) $(SRCS) tb_top.sv tb_main.cpp

vectors:
	$(PYTHON) gen_vectors.py

# per-program run: the vector directory's expect.json gives n_words / n_desc / t_deadlock
run: build
	@test -d vectors/$(PROG) || (echo "no vectors/$(PROG); run make vectors" && exit 1)
	$(eval NWORDS := $(shell $(PYTHON) -c "import json;print(json.load(open('vectors/$(PROG)/expect.json'))['n_words'])"))
	$(eval NDESC  := $(shell $(PYTHON) -c "import json;print(json.load(open('vectors/$(PROG)/expect.json'))['n_desc'])"))
	$(eval TDL    := $(shell $(PYTHON) -c "import json;print(json.load(open('vectors/$(PROG)/expect.json'))['t_deadlock'])"))
	./$(OBJ)/Vtb_top +VEC=vectors/$(PROG) +NWORDS=$(NWORDS) +NDESC=$(NDESC) +TDEADLOCK=$(TDL) +MEMLAT=$(MEMLAT) $(RUNARGS)
	$(PYTHON) compare.py vectors/$(PROG)

all:
	$(MAKE) run PROG=dma_copy
	$(MAKE) run PROG=mma
	$(MAKE) run PROG=vec
	$(MAKE) run PROG=gemm
	$(MAKE) run PROG=fa
	$(MAKE) run PROG=loop
	$(MAKE) run PROG=neg_leak
	$(MAKE) run PROG=neg_deadlock
	$(MAKE) run PROG=neg_untokened TOKEN_CHECK=1

lint:
	$(VERILATOR) --lint-only -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-UNUSEDPARAM -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC --top-module tcta_core $(SRCS)

clean:
	rm -rf obj_dir_tc0 obj_dir_tc1
