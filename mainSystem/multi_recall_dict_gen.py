import pickle

from mainSystem.item_based_recommendation import item_recall_items_dict_gen
from mainSystem.model.data_load import view_load, item_load

from mainSystem.model.text_pro_and_i2i_gen import i2i_sim

from mainSystem.user_based_recommendation import user_recall_items_dict_gen

df_click = view_load()
item_info_df = item_load()

#生成内容emb相似矩阵（数据比较多时开销会很大）
i2i_sim()

#生成召回列表

#user_based
user_recall_items_dict_gen()
#item_based
item_recall_items_dict_gen()

