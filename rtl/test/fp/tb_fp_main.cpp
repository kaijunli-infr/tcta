#include "Vtb_fp.h"
#include "verilated.h"
int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vtb_fp* top = new Vtb_fp; top->clk = 0; top->eval();
    while (!Verilated::gotFinish()) { top->clk = !top->clk; top->eval(); }
    top->final(); delete top; return 0;
}
