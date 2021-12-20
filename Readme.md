### Alert: This probably doesn't work.

# Pathfinder
For CSCE5261: a proof-of-concept actor-critic RL project for predicting variables that achieve timing closure for a given design.

Note that this is my first machine learning project pretty much ever and as such the quality of code is... frankly quite horrendous.

I didn't have much time to train this but the reward trended up. So idk. Maybe this works.

# Neural Network Inputs
| Name | OpenLane Metric |
| - | - |
| Area | `DIEAREA_mm^2` |
| Instance Count | `Total_Physical_Cells` |
| Total IO | `inputs` + `outputs` |
| Net Count | `wire_bits` |
| Deepest Logic Level | `level` |
| Normalized Gate Ratios |`DFF`, `NOT`, `AND`, `NAND`, `OR`, `NOR`, `XOR`, `XNOR`, `MUX` |


# Neural Network Outputs
| OpenLane Variable | Kind |
| - | - |
| {PL,GLB}_RESIZER_SETUP_SLACK_MARGIN | Time (ns) |
| {PL,GLB}_RESIZER_HOLD_SLACK_MARGIN | Time (ns) |
| {PL,GLB}_RESIZER_HOLD_MAX_BUFFER_PERCENT | 0-1 |
| {PL,GLB}_RESIZER_SETUP_MAX_BUFFER_PERCENT | 0-1 |

{PL,GLB}_RESIZER_ALLOW_SETUP_VIOS is always set to `1` and so are {PL,GLB}_PL_RESIZER_TIMING_OPTIMIZATIONS. 

# License
Copyright (c) 2021 The American University in Cairo. Available under the Apache License v2.0.