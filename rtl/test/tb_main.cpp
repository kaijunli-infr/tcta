// Verilator driver: toggles the clock until the testbench calls $finish.
#include "Vtb_top.h"
#include "verilated.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vtb_top* top = new Vtb_top;
    top->clk = 0;
    top->eval();
    while (!Verilated::gotFinish()) {
        top->clk = !top->clk;
        top->eval();
    }
    top->final();
    delete top;
    return 0;
}
