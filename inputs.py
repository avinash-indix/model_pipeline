
import pandas as pd
import numpy as np

accessories_keywords = ["case", "cover", "screen"]
data_size = 3000000
prepare_data  = True



if __name__ == '__main__':

    start = pd.datetime.now()
    # read brands from a list
    df = pd.read_csv("/home/indix/search/bestseller/top500brands_and_sellers.csv");
    # print df.columns
    # print len(df)

    df = df[df['brand'] =='b']
    df['brandText'] = map(lambda b : b.lower().strip(),df['brandText'])
    # print df['brandText']
    print len(df)

    brands = set(df['brandText'])
    print brands
    # assert 1==2
    data_df = pd.read_csv("/home/indix/search/bestseller/model/in_search_with_brand_and_normalizedConfidence.csv",
                          sep='\t')
    print data_df.columns
    print len(data_df)

    data_df['brandname'] = map(lambda bn : bn.lower(),data_df['brandname'])
    # print data_df['brandname'][:50]


    data_df = data_df[data_df['brandname'].isin(brands)]
    print "actual brand data\n"
    print len(data_df)


    data_df.to_csv("/home/indix/search/bestseller/model/brands_in_search_with_brand_and_normalizedConfidence.csv",index=False,sep='\t')
    stop = pd.datetime.now()
    print "Time Taken:\t", stop-start
