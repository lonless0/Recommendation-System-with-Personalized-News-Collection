
import numpy as np
import pandas as pd

def data_gen():
    user_id = np.random.randint(1, 200, size=(2000, 1))

    click_article_id = np.random.randint(1, 1000, size=(2000, 1))
    click_timestamp = np.random.randint(100000, 999999, size=(2000, 1))
    created_at_ts = click_article_id * 333 % 100
    words_count = click_article_id * 137 % 500
    data = np.concatenate(
        [user_id, click_article_id, click_article_id % 8, click_timestamp, created_at_ts, words_count], axis=-1)

    df_click = pd.DataFrame(data,columns=["user_id",'click_article_id','category_id','click_timestamp','created_at_ts','words_count'])

    item_info_df = df_click
    item_info_df = item_info_df.drop(['user_id', 'click_timestamp'], axis=1).drop_duplicates(
        subset=['click_article_id', 'category_id', 'created_at_ts'], keep='first')

    return df_click, item_info_df