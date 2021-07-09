import collections
import pickle

from tqdm import tqdm

from mainSystem.model.data_load import item_load, view_load
from mainSystem.model.data_pre import save_path, item_based_recommend, get_item_topk_click, itemcf_sim, \
    get_item_info_dict, get_user_item_time


def item_recall_items_dict_gen():


    df_click = view_load()
    item_info_df = item_load()

    trn_hist_click_df = df_click

    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(trn_hist_click_df)
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

    i2i_sim = itemcf_sim(df_click, item_created_time_dict)
    i2i_sim = pickle.load(open(save_path + r'\itemcf_i2i_sim.pkl', 'rb'))
    emb_i2i_sim = pickle.load(open(save_path + r'\emb_i2i_sim.pkl', 'rb'))

    sim_item_topk = 20
    recall_item_num = 10
    item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
    for user in tqdm(trn_hist_click_df['user_id'].unique()):
        user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, \
                                                            i2i_sim, sim_item_topk, recall_item_num, \
                                                            item_topk_click, item_created_time_dict, emb_i2i_sim)
    pickle.dump(user_recall_items_dict, open(save_path + r'\itemcf_recall_dict.pkl', 'wb'))


