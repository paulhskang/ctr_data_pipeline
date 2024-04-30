import pandas as pd

# data = [
#     {"frame_id": "bbb", "left_image": "cam1_0001.png", "right_image": "cam1_0001.png"},
#     {"frame_id": "aaa", "left_image": "cam1_0000.png", "right_image": "cam1_0000.png"},
# ]
# frame_ids_set = {"bbb", "aaa"}

data_dict = {
     "bbb": {"left_image": "cam1_0001.png", "right_image": "cam2_0001.png"},
     "aaa": {"left_image": "cam1_0000.png", "right_image": "cam2_0000.png", "another": "11"}
}

df = pd.DataFrame.from_dict(data_dict, orient='index')
df.index.name = "frame_id"
df.to_csv('out.csv')

from_csv_df = pd.read_csv('out.csv',index_col=0)

# print(data)
print(df)
print(from_csv_df)

# print(dict_from_csv)

new_dict = from_csv_df.to_dict(orient='index')
print("old")
print(data_dict)
print("new")
print(new_dict)

# import math
print(pd.isna(new_dict["bbb"]["another"]))
# print(frame_ids_set)
