#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import numpy as np
import pandas as pd
import scipy.sparse
from scipy.sparse import csr_matrix
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
from math import*
from datetime import datetime
import sys
import csv


def square_rooted(x):
 
    return round(sqrt(sum([a*a for a in x])),3)
 
def cosine_similarity(x,y):
 
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def create_long_to_short_names_dict(original_list,shorter_name):
    # Create new names for tags
    new_vals_dict = {}
    for t in enumerate(original_list):
        if t[0] < 10: 
            new_vals_dict[t[1]] = shorter_name + "00" + str(t[0])
        elif (t[0] >= 10) & (t[0] < 100): 
            new_vals_dict[t[1]] = shorter_name + "0" + str(t[0])
        else:
            new_vals_dict[t[1]] = shorter_name + str(t[0])
    return new_vals_dict

def create_short_to_long_names_dict(original_list,shorter_name):
    # Create new names for tags
    new_vals_dict = {}
    for t in enumerate(original_list):
        if t[0] < 10: 
            new_vals_dict[shorter_name + "00" + str(t[0])] = t[1]
        elif (t[0] >= 10) & (t[0] < 100): 
            new_vals_dict[shorter_name + "0" + str(t[0])] = t[1]
        else:
            new_vals_dict[shorter_name + str(t[0])] = t[1]
    return new_vals_dict

def tag_id_lookup(df):
    item_lookup_all = df[["tag_id", "product_name"]]
    item_lookup = item_lookup_all.drop_duplicates()
    return item_lookup

def create_shorter_names_df(df,tags_dict, users_dict):
    df_copy = df.copy()
    df_copy.tag_id = df_copy.tag_id.apply(lambda x: tags_dict[x]) # substitute the tag_id with the one in the dict (shorter)
    df_copy.user_id = df_copy.user_id.apply(lambda x: users_dict[x]) # substitute the user_id with the one in the dict (shorter)
    return df_copy

def create_item_item_df(items_list):
    item_item = pd.DataFrame(0.0, index=items_list, columns=items_list)
    return item_item

def get_user_items_list_df(df):
    tags_by_user = df.groupby("user_id")['tag_id'].agg(list).reset_index()
    return tags_by_user

def get_model(empty_df,tags_by_user,user_item_df):
    list_all_combos = []
    # loop over all users (they are less than the items, so more efficient)
    # for each user go through all combinations of items viewed
    # for each combination calculate the cosine similarity between item vectors and store in the matrix
    for tbu in tags_by_user["tag_id"].to_list():
        for comb in combinations(sorted(tbu),2):
            if comb not in list_all_combos:
                empty_df[comb[0]][comb[1]] = cosine_similarity(list(user_item_df[comb[0]]),list(user_item_df[comb[1]]))
                empty_df[comb[1]][comb[0]] = empty_df[comb[0]][comb[1]]
    return empty_df

def return_top_n_items(tt_similarity_df,tag_id,n,tags_dict,tags_dict_inverse):
    # transform tag_id to shorter version
    tag_id_short = tags_dict[tag_id]
    # extract top n recommended tag_ids
    top_n = list(tt_similarity_df[tt_similarity_df[tag_id_short].isin(sorted(tt_similarity_df[tag_id_short],reverse=True)[:n])].index)
    top_n_tag_id_long = []
    # transform top n tag_ids from short to longer version
    for t in top_n:
        top_n_tag_id_long.append(tags_dict_inverse[t])
    return top_n_tag_id_long


def main(args):
  
  tag_id_to_recommend_upon = args[1]
  n_top_values = int(args[2])
  dataset_path = args[3] # example: "~/Desktop/21_buttons/"
  dataset_name = args[4] # ecample: "21B_tag_views_dataset.csv"

  # Importing data
  print("Importing dataset and creating lookup table...") 
  df = pd.read_csv(dataset_path + dataset_name, sep=",")

  # Select all unique tag_ids
  tags = df.tag_id.unique()
  users = df.user_id.unique()

  # Create new names for tags and put them into dicts for easier reading
  tags_dict = create_long_to_short_names_dict(tags,"t")
  users_dict = create_long_to_short_names_dict(users,"u")
  tags_dict_inverse = create_short_to_long_names_dict(tags,"t")
  users_dict_inverse = create_short_to_long_names_dict(users,"u")

  # Create tag_id - product lookup table
  tag_id_product = tag_id_lookup(df)


  # Get shorter version of tag_id_to_recommend_upon and its product name
  tag_id_to_recommend_upon_short = tags_dict[tag_id_to_recommend_upon]
  tag_id_to_recommend_upon_pr_name = list(tag_id_product[tag_id_product.tag_id.isin([tag_id_to_recommend_upon])]["product_name"])[0]
  print("Recommending over tag_id: {} (or {}), ".format(tag_id_to_recommend_upon,tag_id_to_recommend_upon_short))
  print("product name: {}".format(tag_id_to_recommend_upon_pr_name))

  # Create a copy of the data where we will change the names of the tags to make it simpler to read
  df_copy = create_shorter_names_df(df,tags_dict,users_dict)

  # Filter the input dataframe with only users who viewed the tag_id_to_recommend_upon
  users_viewed_tag_id = list(df_copy[df_copy.tag_id == tag_id_to_recommend_upon_short]["user_id"])
  print("There are {} users who viewed the inputed tag_id".format(len(list(users_viewed_tag_id))))
  filtered_df = df_copy[df_copy.user_id.isin(users_viewed_tag_id)]

  # Create a df with user_id and for each the list of tags viewed
  tags_by_user = get_user_items_list_df(filtered_df)
  # Create empty dataset (with all zeros) for tags vs tags (item-item)
  tt_similarity_df = create_item_item_df(filtered_df.tag_id.unique())
  # Create user-item df to calculate cosine similarity later
  user_item_df = pd.crosstab(filtered_df.user_id, filtered_df.tag_id)

  # Run model (item-item similarity matrix)
  print("Creating item-item similarity matrix based on only users who have viewed that tag_id...")
  start_time = datetime.now()
  tt_similarity_df = get_model(tt_similarity_df,tags_by_user,user_item_df)
  time_elapsed = datetime.now() - start_time 
  print("...done!")
  print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

  # Extract top-10 recommendations for given item
  top_10 = return_top_n_items(tt_similarity_df,tag_id_to_recommend_upon,n_top_values,tags_dict,tags_dict_inverse)

  top_10_product_names = []
  for t in top_10:
      pr_name = list(tag_id_product[tag_id_product.tag_id == t]["product_name"])[0]
      top_10_product_names.append(pr_name)
      
  print("For input tag_id: {},".format(tag_id_to_recommend_upon))
  print("With product name: {} \n".format(list(tag_id_product[tag_id_product.tag_id == tag_id_to_recommend_upon]["product_name"])[0]))
  print("Tag_ids are: \n {}".format(top_10))
  print("Corresponding to products: \n {}".format(top_10_product_names))


if __name__=='__main__':
  
  args = sys.argv
  num_args = 4
  
  if len(args) != num_args+1:
    print('Parameter error: {} expected, {} found.'.format(num_args, len(args)-1))
    print('Usage: {} <tag_id_to_recommend_upon> <n_top_values> <dataset_path> <dataset_name>'.format(args[0]))
    sys.exit(-1)
    
  main(args) 