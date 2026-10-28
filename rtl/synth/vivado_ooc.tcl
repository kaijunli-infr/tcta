# Vivado out-of-context synthesis + implementation of tcta_core for the KV260 (xck26-sfvc784-2LV-c).
#   vivado -mode batch -source rtl/synth/vivado_ooc.tcl          (x86 machine with Vivado 2023.2+; run from the repo root)
# Writes rtl/synth/vivado/{utilization,timing}.rpt. The target clock is 200 MHz (vivado_ooc.xdc); the report's
# WNS tells how far the single-cycle FP32 paths are from it.
set root [file normalize [file dirname [info script]]/../..]
set out  $root/rtl/synth/vivado
file mkdir $out
create_project -in_memory -part xck26-sfvc784-2LV-c
set_property target_language Verilog [current_project]
set srcs [list tcta_isa_pkg.sv tcta_pkg.sv fp/f32_add.sv fp/f32_mul.sv fp/f32_misc.sv sfu.sv token_fifo.sv token_net.sv \
               scratchpad.sv tile_meta.sv token_check.sv dequant.sv dma_load.sv dma_store.sv mma_pe.sv mma_unit.sv \
               vec_unit.sv front_end.sv queue.sv tcta_core.sv]
foreach f $srcs { read_verilog -sv $root/rtl/src/$f }
read_xdc $root/rtl/synth/vivado_ooc.xdc
synth_design -top tcta_core -mode out_of_context -flatten_hierarchy rebuilt -directive AreaOptimized_high
report_utilization -hierarchical -file $out/utilization_synth.rpt
opt_design
place_design -directive Explore
phys_opt_design
route_design -directive Explore
report_utilization -hierarchical -file $out/utilization.rpt
report_timing_summary -max_paths 20 -file $out/timing.rpt
report_timing -sort_by slack -max_paths 10 -file $out/worst_paths.rpt
write_checkpoint -force $out/tcta_core_routed.dcp
