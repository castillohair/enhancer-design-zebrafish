from pathlib import Path

import pandas

# Stage / cell state info
#########################
# Load from resources/cell_state_metadata.csv
cell_state_metadata_filepath = Path(__file__).resolve().parent / 'resources' / 'cell_state_metadata.csv'
cell_state_metadata_df = pandas.read_csv(cell_state_metadata_filepath)

CELL_STATES = cell_state_metadata_df['cell_state'].tolist()
STAGES = cell_state_metadata_df['stage'].drop_duplicates().tolist()

# DeepDanio definitions
#######################
DEEPDANIO_MODEL_PATH = {
    0: 'models/deepdanio/deepdanio_data_split_0.h5',
    1: 'models/deepdanio/deepdanio_data_split_1.h5',
    2: 'models/deepdanio/deepdanio_data_split_2.h5',
}
DEEPDANIO_INPUT_LENGTH = 500
