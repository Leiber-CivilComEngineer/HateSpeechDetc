# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#
import pandas as pd
from sklearn.model_selection import train_test_split

def split_df(df, train_ratio, develop_ratio, test_ratio, random_state=1):
    if train_ratio+develop_ratio+test_ratio != 1:
        raise ValueError("ratios did not add up to 1")
    if develop_ratio == 0:
        train_df, test_df = train_test_split(df, train_size=train_ratio, test_size=test_ratio, random_state=random_state)
        return train_df, test_df
    else:
        train_df, remaining_df = train_test_split(df, train_size=train_ratio, test_size=1-train_ratio, random_state=random_state)
        develop_relative_ratio = develop_ratio/(develop_ratio+test_ratio)
        develop_df, test_df = train_test_split(remaining_df, train_size=develop_relative_ratio, test_size=1-develop_relative_ratio, random_state=random_state)
        return train_df, develop_df, test_df