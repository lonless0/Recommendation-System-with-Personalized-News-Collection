import collections
import pickle
from tqdm import tqdm
from mainSystem.model.data_gen import data_gen
from mainSystem.model.data_pre import max_min_scaler, get_user_item_time, get_item_topk_click, get_item_info_dict, \
    get_user_activate_degree_dict, usercf_sim, save_path, user_based_recommend
from model.text_pro_and_i2i_gen import i2i_sim
from model.data_load import view_load, item_load


def user_recall_items_dict_gen():
    df_click = view_load()
    item_info_df = item_load()

    user_activate_degree_dict = get_user_activate_degree_dict(df_click)
    trn_hist_click_df = df_click
    user_recall_items_dict = collections.defaultdict(dict)
    user_item_time_dict = get_user_item_time(trn_hist_click_df)
    #生成用户相似矩阵
    u2u_sim = usercf_sim(df_click, user_activate_degree_dict)
    #读取用户相似矩阵
    u2u_sim = pickle.load(open(save_path + r'\usercf_u2u_sim.pkl', 'rb'))

    sim_user_topk = 20
    recall_item_num = 10
    item_topk_click = get_item_topk_click(trn_hist_click_df, k=50)
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

    #生成emb相似矩阵
    #emb_i2i_sim = i2i_sim()
    emb_i2i_sim_ = pickle.load(open(save_path+r'\emb_i2i_sim.pkl', 'rb'))




    for user in tqdm(trn_hist_click_df['user_id'].unique()):
        user_recall_items_dict[user] = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk, \
                                                            recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim_)
    pickle.dump(user_recall_items_dict, open(save_path + r'\usercf_u2u2i_recall.pkl', 'wb'))


#生成测试数据
#df_click, item_info_df = data_gen()
#正式使用时使用该部分导入数据





#生成召回列表
#user_recall_items_dict_gen()

#读取召回列表
# user_recall_items_dict = pickle.load(open(save_path + 'usercf_u2u2i_recall.pkl', 'rb'))

# test_user_id = 11
# print(test_user_id,':',user_recall_items_dict[test_user_id])
# print(user_recall_items_dict)


