#!/bin/bash

#source activate "dl2-project"

# State: {h_j}, without {distance, PE init}
python main.py --pe nope --write_config_to mpnn_1

# State: {h_j} with PE init, without {distance}
python main.py --pe rw --pe_dim 24 --write_config_to mpnn_2

# State: {h_j, distance} without {PE init}
python main.py --pe nope --include_dist --write_config_to mpnn_3

# State: {h_j, distance} with PE init
python main.py --pe rw --pe_dim 24 --include_dist --write_config_to mpnn_4

# State: {h_j} with PE init and LSPE, without {distance}
python main.py --pe rw --pe_dim 24 --lspe --write_config_to mpnn_5

# State: {h_j, distance} with PE init and LSPE
python main.py --pe rw --pe_dim 24 --include_dist --lspe --write_config_to mpnn_6

# State: {h_j}, without {distance, PE init}
python main.py --pe nope --write_config_to mpnn_7 --reduced

# State: {h_j} with PE init, without {distance}
python main.py --pe rw --pe_dim 24 --write_config_to mpnn_8 --reduced

# State: {h_j, distance} without {PE init}
python main.py --pe nope --include_dist --write_config_to mpnn_9 --reduced

# State: {h_j, distance} with PE init
python main.py --pe rw --pe_dim 24 --include_dist --write_config_to mpnn_10 --reduced

# State: {h_j} with PE init and LSPE, without {distance}
python main.py --pe rw --pe_dim 24 --lspe --write_config_to mpnn_11 --reduced

# State: {h_j, distance} with PE init and LSPE
python main.py --pe rw --pe_dim 24 --include_dist --lspe --write_config_to mpnn_12 --reduced