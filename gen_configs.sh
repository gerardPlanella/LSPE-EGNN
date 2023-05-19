#!/bin/bash

#source activate "egnn_lspe_gpu"



# # State: {h_j}, without {distance, PE init}
# python main.py --pe nope --write_config_to mpnn_1

# # State: {h_j} with PE init, without {distance}
# python main.py --pe rw --pe_dim 24 --write_config_to mpnn_2

# # State: {h_j, distance} without {PE init}
# python main.py --pe nope --include_dist --write_config_to mpnn_3

# # State: {h_j, distance} with PE init
# python main.py --pe rw --pe_dim 24 --include_dist --write_config_to mpnn_4

# # State: {h_j} with PE init and LSPE, without {distance}
# python main.py --pe rw --pe_dim 24 --lspe --write_config_to mpnn_5

# # State: {h_j, distance} with PE init and LSPE
# python main.py --pe rw --pe_dim 24 --include_dist --lspe --write_config_to mpnn_6

# # State: {h_j}, without {distance, PE init}
# python main.py --pe nope --write_config_to mpnn_7 --reduced

# # State: {h_j} with PE init, without {distance}
# python main.py --pe rw --pe_dim 24 --write_config_to mpnn_8 --reduced

# # State: {h_j, distance} without {PE init}
# python main.py --pe nope --include_dist --write_config_to mpnn_9 --reduced

# # State: {h_j, distance} with PE init
# python main.py --pe rw --pe_dim 24 --include_dist --write_config_to mpnn_10 --reduced

# # State: {h_j} with PE init and LSPE, without {distance}
# python main.py --pe rw --pe_dim 24 --lspe --write_config_to mpnn_11 --reduced

# # State: {h_j, distance} with PE init and LSPE
# python main.py --pe rw --pe_dim 24 --include_dist --lspe --write_config_to mpnn_12 --reduced

# MPNN-LSPE - include PE in update
python main.py --model mpnn --pe rw --pe_dim 24 --lspe --update_with_pe --write_config_to mpnn_14

# MPNN-LSPE-Geom - include PE in update
python main.py --model mpnn --pe rw --pe_dim 24 --lspe --include_dist --update_with_pe --write_config_to mpnn_15

# MPNN-Reduced-LSPE - include PE in update
python main.py --model mpnn --pe rw --pe_dim 24 --lspe --update_with_pe --reduced --write_config_to mpnn_16

# MPNN-LSPE-Geom - include PE in update
python main.py --model mpnn --pe rw --pe_dim 24 --lspe --include_dist --update_with_pe --reduced --write_config_to mpnn_17

# MPNN-Geom-7layers-yes_fc_nope_yesdist-no lspe
python main.py --model mpnn --include_dist --num_layers 7 --dataset qm9_fc --pe nope --write_config_to mpnn_18

# MPNN-Geom-7layers-yes_fc_rwope_yesdist-no lspe
python main.py --model mpnn --include_dist --num_layers 7 --dataset qm9_fc --pe rw --pe_dim 24 --write_config_to mpnn_19


# #MPNN-Geom-fc 7 layers, fc
# python main.py --model mpnn --include_dist --dataset qm9_fc --num_layers 7 --write_config_to mpnn_18

# #MPNN-Geom-fc 4 layers, fc
# python main.py --model mpnn --include_dist --dataset qm9_fc --num_layers 4 --write_config_to mpnn_19

# #MPNN-Geom-fc 4 layer, no fc
# python main.py --model mpnn --include_dist --dataset qm9 --num_layers 4 --write_config_to mpnn_20

# #MPNN-Geom-fc 1 layer, fc
# python main.py --model mpnn --include_dist --dataset qm9_fc --num_layers 1 --write_config_to mpnn_21

# #MPNN-Geom-fc 1 layer, no fc
# python main.py --model mpnn --include_dist --dataset qm9 --num_layers 1 --write_config_to mpnn_22
