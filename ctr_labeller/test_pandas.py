import pandas as pd

data = [
    {"frame_id": "aaa", "left_image": "cam1_0000.png", "right_image": "cam1_0000.png"},
    {"frame_id": "bbb", "left_image": "cam1_0001.png", "right_image": "cam1_0001.png"}
]
print(data)
frame_ids_set = {"aaa", "bbb"}
print("ccc" in frame_ids_set)
print("aaa" in frame_ids_set)


df = pd.DataFrame.from_dict(data)
print(df)

df.to_csv('out.csv', index=False)

from_csv_df = pd.read_csv('out.csv')
dict_from_csv = from_csv_df.to_dict('records')
print(df)
print(dict_from_csv)

# Convert to set
frame_ids_set = set(from_csv_df["frame_id"])
print(frame_ids_set)
