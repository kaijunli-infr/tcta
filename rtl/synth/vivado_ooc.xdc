# 200 MHz target for the out-of-context run (ISA §11 KV260 column)
create_clock -period 5.000 -name clk [get_ports clk]
set_input_delay  -clock clk 0.5 [all_inputs]
set_output_delay -clock clk 0.5 [all_outputs]
