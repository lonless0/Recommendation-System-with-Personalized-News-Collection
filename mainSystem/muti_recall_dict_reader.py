#读取召回列表
import pickle

from mainSystem.model.data_pre import save_path

user_multi_recall_dict =  {'item_based': {},
                           'user_based': {},
                           'cold_start_recall': {}}


user_multi_recall_dict['user_based'] = pickle.load(open(save_path + r'/usercf_u2u2i_recall.pkl', 'rb'))
user_multi_recall_dict['item_based'] = pickle.load(open(save_path + r'/itemcf_recall_dict.pkl', 'rb'))

#print(test_user_id,':',user_recall_items_dict[test_user_id])
#print(user_recall_items_dict)




print(user_multi_recall_dict['user_based'])
print(user_multi_recall_dict['item_based'])