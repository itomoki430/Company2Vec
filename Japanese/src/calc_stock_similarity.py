import pandas as pd
import pickle

revised_month_stock_df_dict = pickle.load(open("../dataset/revised_month_stock_df_dict.pkl","rb"))
use_ticker_list_sec = []
use_stock_price_list = []
sector_array = []
industry_array = []
use_text_data = []
stock_size = 58
use_ticker_list_limit = []
use_stock_price_list_limit  = []
use_sec_code_list = []
for index, ticker in enumerate(use_ticker_list):
    sec_code = int(str(ticker)[:4])
    use_sec_code_list.append(sec_code)
    if sec_code in sec_code2sector:
        if len(revised_month_stock_df_dict[ticker]["return"][-stock_size:]) >= 12:
                if  (type(sec_code2sector[sec_code ]) == str) :
                    use_stock_price_list.append(list(revised_month_stock_df_dict[ticker]["return"])[-stock_size:])
                    sector_array.append(sec_code2sector[sec_code])
                    use_text_data.append(text_data[sec_code])
                    industry_array.append(sec_code2industry[sec_code])
                    use_ticker_list_sec.append(sec_code)
        if len(revised_month_stock_df_dict[ticker]["return"]) == stock_size:
            use_ticker_list_limit.append(sec_code)
            use_stock_price_list_limit.append(list(revised_month_stock_df_dict[ticker]["return"]))            
    #else:
        #print (sec_code)

stock_similarity_mat_limit = pd.read_csv("stock_data/common_length_mat.csv", index_col=0)
#use_ticker_list_limit_pair_cossim = sklearn.metrics.pairwise.cosine_similarity(np.array(use_stock_price_list_limit))
#stock_similarity_mat_limit = pd.DataFrame(dict(zip(use_ticker_list_limit, use_ticker_list_limit_pair_cossim)))
#stock_similarity_mat_limit.index = use_ticker_list_limit
#stock_similarity_mat_limit.to_csv("stock_data/common_length_mat.csv")

def calc_stockprice_cosine_similarity(code_1, code_2):
    #moxnth_stock_info_df_1 = month_stock_info_df[month_stock_info_df["ticker"] == code_1]
    #month_stock_info_df_2 = month_stock_info_df[month_stock_info_df["ticker"] == code_2]
    month_stock_info_df_1 = revised_month_stock_df_dict[code_1]
    month_stock_info_df_2 = revised_month_stock_df_dict[code_2]
    common_months  = (set(month_stock_info_df_1["month"]) &  set(month_stock_info_df_2["month"]))
    #print (len(common_months))
    if len(common_months)  < 12:
        return 0
    resurn_1 = month_stock_info_df_1[[(month in common_months) for month in month_stock_info_df_1["month"]]]["return"]
    resurn_2 = month_stock_info_df_2[[(month in common_months) for month in month_stock_info_df_2["month"]]]["return"]
    resurn_1_norm = np.array(resurn_1)/np.linalg.norm(resurn_1)
    resurn_2_norm = np.array(resurn_2)/np.linalg.norm(resurn_2)
    #return sklearn.metrics.pairwise.cosine_similarity(np.array([resurn_1_norm, resurn_2_norm]))
    #return sklearn.metrics.pairwise.cosine_similarity(np.array([resurn_1, resurn_2]))
    return resurn_1_norm.dot(resurn_2_norm)

remain_tick_list = np.sort(list(set(use_sec_code_list) - set(use_ticker_list_limit)))


additional_similarity_mat = []
for index_1, ticker_1 in enumerate(remain_tick_list):
    df = pd.read_csv("stock_data/stock_similarity_mat/" + str(ticker_1) + ".csv", index_col=1)
    df = df.T[1:].T
    df.columns = [ ticker_1]
    additional_similarity_mat.append(df)

additional_similarity_mat_df = pd.concat(additional_similarity_mat, axis = 1).T
additional_similarity_mat_df.columns = [int(str(val)[:4]) for val in additional_similarity_mat_df.columns]
common_similarity_mat_df = pd.read_csv("stock_data/common_length_mat.csv",  index_col=0)
common_similarity_mat_df.columns = np.array(common_similarity_mat_df.columns).astype(int)


similarity_mat_df = pd.concat([
    pd.concat([
    pd.concat([common_similarity_mat_df[index], additional_similarity_mat_df[index]]).sort_index()
    for index in common_similarity_mat_df.index], axis =1), 
      additional_similarity_mat_df.T], axis = 1).T.sort_index()


similarity_mat_df.to_csv("stock_data/all_similarity_mat.csv")