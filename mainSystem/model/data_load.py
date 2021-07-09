import pandas as pd
from mainSystem.model.data_pre import max_min_scaler
from sklearn.preprocessing import LabelEncoder


def view_load():
    label = LabelEncoder()
    df_click = pd.read_csv(r'C:\Users\lonless\Documents\GitHub\Recommendation-System-with-Personalized-News-Collection\mainSystem\data\view.csv')
    df_click['click_timestamp'] = df_click[['click_timestamp']].apply(max_min_scaler())
    df_click['created_at_ts'] = df_click[['created_at_ts']].apply(max_min_scaler())
    df_click['category_id'] = label.fit_transform(df_click['category_id'])

    return df_click


def item_load():
    label = LabelEncoder()
    item_info_df = pd.read_csv(r'C:\Users\lonless\Documents\GitHub\Recommendation-System-with-Personalized-News-Collection\mainSystem\data\news.csv')
    item_info_df['created_at_ts'] = item_info_df[['created_at_ts']].apply(max_min_scaler())
    item_info_df['category_id'] = label.fit_transform(item_info_df['category_id'])

    return item_info_df
