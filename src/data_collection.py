import pandas as pd
import os

def store_data(speed_data, output_folder):
    speed_df = pd.DataFrame(speed_data, columns=['Speed'])
    speed_df.to_csv(os.path.join(output_folder, "speed_data.csv"), index=False)
