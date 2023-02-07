import re
from datetime import datetime as dt
from calendar import isleap
import pandas as pd
import numpy as np
from functools import reduce 
import sympy as sym

import concurrent.futures
import GDrive as gd

SEC_DIR = 'Data/Securities/'
SEC_MAP_PATH = 'Data/Search_Indices/GDrive_Securities_index.csv'
SEC_MAP_NAME = 'GDrive_Securities_index.csv'
SEC_MAP_ID = '1eZ4FkRcW2JwS6i1LST5U4jqTRtQk701p' # id of the file mapping {name:id} for the folder 'Data/Securities'

# Select/Find securities
if True:
    def get_cloud_sec_map(cloud_map_id=SEC_MAP_ID, service=None):
        cloud_map = gd.get_GDrive_map_from_id(cloud_map_id, service=service)
        return cloud_map

    def select_securities(ticker=None, ticker_and_letter=None, include_continuous=False, cloud_map_id=None, cloud_map_dict=None, service=None):
        folder=SEC_DIR
        if cloud_map_dict==None:
            cloud_map_id=SEC_MAP_ID

        all_files=gd.listdir(folder, cloud_map_id=cloud_map_id, cloud_map_dict=cloud_map_dict, service=service)

        if ticker is not None:            
            fo = [sec for sec in all_files if info_ticker(sec)==ticker]
        elif ticker_and_letter is not None:
            fo = [sec for sec in all_files if info_ticker_and_letter(sec)==ticker_and_letter]
        else:
            fo = all_files


        # Filter by continuous
        if not include_continuous:
            fo = [sec for sec in fo if (not is_continuous(sec))]

        fo = [sec.replace('.csv','') for sec in fo]
        return fo

    def select_funds():
        all_files = gd.listdir(SEC_DIR)
        fo =[f.split('_')[0] for f in all_files if 'funds' in f]
        return fo

# Read securities, Calendars, VaR files, etc...
if True:
    def read_security_list(sec_list=[], parallel=None, max_workers=500, cloud_map_dict=None, service=None, cloud=False):
        fo={}
        if cloud:
            parallel='thread'

        if parallel is None:
            for sec in sec_list:
                fo[sec] = read_security(sec, cloud, service)

        elif parallel=='thread':
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for sec in sec_list:
                    results[sec] = executor.submit(read_security, sec, cloud, service)
            
            for key, res in results.items():
                fo[key]=res.result()

        elif parallel=='process':
            with concurrent.futures.ProcessPoolExecutor(max_workers= min(max_workers,61)) as executor:
                results={}
                for sec in sec_list:
                    results[sec] = executor.submit(read_security, sec, cloud, service)
            
            for key, res in results.items():
                fo[key]=res.result()

        if gd.is_cloud_id(sec_list[0]) and (cloud_map_dict is not None):
            cloud_map_dict = {v: k for k, v in cloud_map_dict.items()} # reverse the dictionary to {id:file_name}
            fo= {cloud_map_dict[id].replace('.csv','') :df for id, df in fo.items()}
                
        return fo
    
    def read_security(sec, cloud=False, service=None, check_cloud_id = False):
        if check_cloud_id and gd.is_cloud_id(sec):
            file=sec
        else:
            if '_' not in sec:
                sec=sec+'_0'
            if '.csv' not in sec:
                sec=sec+'.csv'

        if cloud:            
            file=sec # in this way it goes faster (as it doesn't have to request for the folders in the path)
        else:
            file = SEC_DIR + sec

        df = gd.read_csv(file_path=file, service=service, parse_dates=['date'], dayfirst=True, index_col='date')

        return df

    def read_calendar(service=None):
        file = SEC_DIR+ 'all_calendar.csv'
        df=gd.read_csv(file,parse_dates=['start','end'], dayfirst=True, index_col='security', service=service)
        return df
    
    def read_dates(service=None):
        file = SEC_DIR+ 'all_dates.csv'
        df=gd.read_csv(file,parse_dates=['first_trade_date', 'first_notice_date', 'first_delivery_date', 'last_delivery_date', 'last_trade_date'], dayfirst=True, index_col='security', service=service)
        return df
# Securities elaborations
if True:
    def calc_all_volatilities(df):
        df=calc_volatility(df, vol_to_calc='implied_vol_dm', min_vol=0, max_vol=150, max_daily_ratio_move=2.0, holes_ratio_limit=1.2)
        df=calc_volatility(df, vol_to_calc='implied_vol_dm_call_25d', min_vol=0, max_vol=150, max_daily_ratio_move=2.0, holes_ratio_limit=1.2)
        df=calc_volatility(df, vol_to_calc='implied_vol_dm_put_25d', min_vol=0, max_vol=150, max_daily_ratio_move=2.0, holes_ratio_limit=1.2)                
        return df

    def calc_volatility(df, vol_to_calc='implied_vol_dm', min_vol=0, max_vol=150, max_daily_ratio_move=2.0, holes_ratio_limit=1.2):
        '''
            the below to calculate the 50 delta
            vol_couples=[                 
                            ['implied_vol_call_50d', 'implied_vol_put_50d'],
                            ['implied_vol_hist_put', 'implied_vol_hist_call'],        
                            ['bvol_50d'],
                        ]

            min_vol=0
            max_vol=150

            max_daily_ratio_move=2.0 
                    - if the volatility moves more than double (or more than halve): discard
            
            holes_ratio_limit=1.2

                    - the above 'holes_ratio_limit' is to reject 'data holes': a series that goes (12, 14, 13, 15, 0, 16, 17) clearly shows that the '0' is wrong
                     so all the left and right ratios are calculated, and when they are bigger/smaller than 1.2, the '0' is replaced by the average of left and right values                    
        '''

        if vol_to_calc=='implied_vol_dm':
            vol_couples=[ ['implied_vol_call_50d', 'implied_vol_put_50d'], ['implied_vol_hist_put', 'implied_vol_hist_call'], ['bvol_50d']]

        if vol_to_calc=='implied_vol_dm_call_25d':
            vol_couples=[ ['implied_vol_call_25d', 'implied_vol_call_25d'], ['bvol_call_25d', 'bvol_call_25d']]

        if vol_to_calc=='implied_vol_dm_put_25d':
            vol_couples=[ ['implied_vol_put_25d', 'implied_vol_put_25d'], ['bvol_put_25d', 'bvol_put_25d']]

        vol_cols = sum(vol_couples, [])
        df_vol=df[vol_cols]

        mask=((df_vol<min_vol) & (max_vol<df_vol)) # removing values less than 0 and higher than 150%
        df_vol[mask]=np.nan
        new_cols=[] # one new column for each 'couple' and they are going to be called 0,1,2,etc 

        for i, cols_couple in enumerate(vol_couples):
            df_vol[i]=df_vol[cols_couple].mean(skipna=True, axis=1)
            new_cols.append(i)

        fvi=df_vol.first_valid_index()
        lvi=df_vol.last_valid_index()

        if fvi is None:
            return df
        
        if lvi is None:
            return df                

        mask=((df_vol.index>=fvi) & (df_vol.index<=lvi))
        ids= df_vol[mask].index # these are all the indices that will need to be filled


        passes=['forward','backward']

        for p in passes:
            # Reverse the index for the 'backward' pass
            if p=='backward':
                ids=ids.sort_values(ascending=False)

            # Initializing to NaN
            df_vol[p]=np.nan

            # Fill the first value
            first_row=(df_vol.loc[ids[0]][new_cols])
            mask=first_row.notna()
            df_vol.loc[ids[0]][p] = first_row[mask].iloc[0] # by picking iloc[0] we are sure to get the first available with the priority assigned to the 'couples'

            # Initialize the 'previous' for the following calc
            previous=df_vol.loc[ids[0]][p]

            # Filling everything else
            for i in range(1,len(ids)):
                row = df_vol.loc[ids[i]][new_cols] # full df row

                list_diff= abs(row - previous)
                list_diff= list_diff[list_diff.notna()]
                if len(list_diff)==0:
                    continue

                # if there are competing values, pick the one with the smallest daily change
                sel_col = list_diff.idxmin()

                # check if the selected column has an acceptable ratio change, because above I only selected the smallest change, I didn't check if it makes sense (maybe they are ALL wrong)
                list_ratio= abs(row/previous)

                if (list_ratio[sel_col]>max_daily_ratio_move) or (list_ratio[sel_col]<(1.0/max_daily_ratio_move)):
                    continue

                df_vol.loc[ids[i]][p]=row[sel_col]
                previous=row[sel_col]

        # Counting the valid values and select the best
        valid_values = df_vol[passes].notna().sum()
        if valid_values['forward']>=valid_values['backward']:
            df_vol[vol_to_calc]=df_vol['forward'] 
        else:
            df_vol[vol_to_calc]=df_vol['backward']


        # Filling the holes (by changing the crazy values, so I start filtering the NaN)
        mask=(df_vol[vol_to_calc].notna())
        ids = df_vol.loc[mask].index

        for i in range(1,len(ids)-1):
            left_value=df_vol[vol_to_calc][i-1]
            right_value=df_vol[vol_to_calc][i+1]

            ratio_left = df_vol[vol_to_calc][i] / left_value
            ratio_right = df_vol[vol_to_calc][i] / right_value

            if ((ratio_left > holes_ratio_limit) and (ratio_right > holes_ratio_limit)):
                df_vol[vol_to_calc][i]=(left_value+right_value)/2.0

            if ((ratio_left < (1 / holes_ratio_limit)) and (ratio_right < (1 / holes_ratio_limit))):
                df_vol[vol_to_calc][i]=(left_value+right_value)/2.0

        df[vol_to_calc]=df_vol[vol_to_calc]
        return df

    def sec_dfs_simple_sec(sec_dfs):
        '''
        sec_dfs = {'w n_2020':df}
        '''
        fo={}
        sel_sec=[]
        for sec, d in sec_dfs.items():
            if (info_type(sec)!='future') and (info_maturity(sec)==0):
                sel_sec.append(sec)

        for sec in sel_sec:
            ticker=info_ticker(sec)
            for y in set(sec_dfs[sec].index.year):
                sec_dfs[ticker+'_'+str(y)]=sec_dfs[sec]
            del sec_dfs[sec]
            
        return sec_dfs

    def create_seas_df(expression, sec_dfs, col='close_price', ref_year=None, seas_interval=None):
        '''
        sec_dfs = {'w n_2020':df}
        '''
        if ref_year is None:
            ref_year=dt.today().year
        if seas_interval is None:
            seas_interval=[dt.today()-pd.DateOffset(months=6)+pd.DateOffset(days=1), dt.today()+pd.DateOffset(months=6)]
        
        symbols=extract_symbols_from_expression(expression)
        
        df_list=[]
        for symb in symbols:
            dfs=[]
            for sec, d in sec_dfs.items():
                if info_ticker_and_letter(sec)==symbol_no_offset(symb):
                    df=d[:]
                    year=info_maturity(sec).year-symbol_offset(symb)
                    offset = ref_year-year
                    interval= [i - pd.DateOffset(years=offset) for i in seas_interval]

                    df['sec']=sec
                    df['year']=year

                    mask=(df.index>=interval[0]) & (df.index<=interval[1])
                    # df=df[mask][['year','sec',col]]
                    df=df[mask][['year',col]]
                    df=add_seas_timeline(df, offset)
                    
                    dfs.append(df)

            df=pd.concat(dfs)
            df=df.rename(columns={col:symb})
            df_list.append(df)
        
        df = reduce(lambda left, right: pd.merge(left , right,on = ["seas_day", "year"], how = "outer"),df_list)
        
        df[col]=evaluate_expression(df,expression)
        df=df.pivot(index='seas_day',columns='year',values=col)  
        df=df.interpolate(method='polynomial', order=0, limit_area='inside')

        return df

    def seas_avg(df):
        return df
# get Info
if True:
    def info_type(sec):
        
        if is_fx(sec):
            return 'fx'
        elif 'fund' in sec:
            return 'fund'
        else:
            return 'future'

    def info_ticker_and_letter(sec):
        split=sec.split('_')
        
        if len(split)==1:
            # there is no '_' in the security
            return sec
        
        if split[-1]=='0':
            return split[-2]
        
        else:
            return split[-2]
        
    def info_ticker(sec):
        split=sec.split('_')
        
        if len(split)==1:
            # there is no '_' in the security
            return split[0]
        if split[-1]=='0':
            return split[-2]
        if is_fx(sec):
            return split[0]
        if 'fund' in sec:
            return split[0]
        else:
            return split[-2][0:-1]

    def info_maturity(sec):
        split=sec.split('_')
        if info_type(sec)=='future':
            year= int(split[1])
            letter= split[0][-1]
            month=month_from_letter(letter)
            return dt(year,month,1)
        else:
            year= int(split[1])
            if year==0:
                return 0
            return dt(year,1,1)
    
    def is_continuous(sec):
        letter=info_ticker_and_letter(sec)[-1]
        if letter=='1' or letter=='2':
            return True            
        return False
    
    def is_fx(sec):
        fx=['eur','usd','gbp', 'chf']

        for f in fx:
            if f in sec:
                return True
                     
        return False


# Accessories
if True:
    def add_seas_timeline(df, years_offset, date_col=None):
        '''
        timeline:
            - original time line
        fo:
            - the 'common' seasonal timeline (the x-axis on the final 'seas' chart)
        '''
        fo = []
        if date_col==None:
            timeline=df.index
        else:
            timeline=df[date_col]

        for d in timeline:
            # print('timeline:', timeline)
            year = d.year + years_offset
            month = d.month
            day = d.day

            if (month == 2) and (day == 29) and (not isleap(year)):
                fo.append(dt(year, month, 28, 12,00,00))
            else:
                fo.append(dt(year, month, day))

        df['seas_day'] = fo
        return df

    def month_from_letter(letter):
        if letter=='f':
            return 1
        elif letter=='g':
            return 2
        elif letter=='h':
            return 3
        elif letter=='j':
            return 4
        elif letter=='k':
            return 5
        elif letter=='m':
            return 6
        elif letter=='n':
            return 7
        elif letter=='q':
            return 8
        elif letter=='u':
            return 9
        elif letter=='v':
            return 10
        elif letter=='x':
            return 11
        elif letter=='z':
            return 12

    def letter_from_month(month):
        if month==1:
            return 'f'
        elif month==2:
            return 'g'
        elif month==3:
            return 'h'
        elif month==4:
            return 'j'
        elif month==5:
            return 'k'
        elif month==6:
            return 'm'
        elif month==7:
            return 'n'
        elif month==8:
            return 'q'
        elif month==9:
            return 'u'
        elif month==10:
            return 'v'
        elif month==11:
            return 'x'
        elif month==12:
            return 'z'  

    def relative_security(security, base_year):
        # 'security' must be in the form 'c z_2022'
        split=security.split('_')

        ticker=split[0][0:-1]
        letter=split[0][-1]    
        relative_year=int(split[1])-base_year

        return ticker+letter+str(relative_year)


# Symbolic Expressions
if True:
    def symbol_no_offset(symbol):
        offset_str=symbol[-1]
        if offset_str.isnumeric():
            return symbol[0:-1]
        else:
            return symbol
        
    def symbol_offset(symbol):
        offset_str=symbol[-1]
        if offset_str.isnumeric():
            return int(offset_str)
        else:
            return 0
        
    def dm_split(string, separators = "-+*/()'^."):
        result = re.split('|'.join(map(re.escape, separators)), string)
        return result

    def dm_replace(string, args_dict={}):
        for k, v in args_dict.items():
            string = string.replace(k, v)
        return string
    
    def dm_isnumeric(string):
        try:
            float(string)
            return True
        except ValueError:
            return False


    def extract_symbols_from_expression(expression):
        # the symbolic package doesn't like spaces in symbols
        # so this function returns a dictionary {original:modified}
        separators = "-+*/()^"
        fo=dm_split(expression,separators)
        fo=[s.strip() for s in fo]
        # fo=[s for s in fo if not s.isnumeric()]
        fo=[s for s in fo if not dm_isnumeric(s)]
        
        fo={s: s.replace(' ','_') for s in fo}
        if '' in fo:
            del fo['']

        return fo

    def evaluate_expression(df,expression):
        symbols_dict=extract_symbols_from_expression(expression)
        expression=dm_replace(expression, symbols_dict)

        symbols = sym.symbols(list(symbols_dict.values()))
        
        expression = sym.sympify(expression)
        f = sym.lambdify([symbols], expression, 'numpy')

        cols=list(symbols_dict.keys())
        # var_list=[df[c] for c in cols] # preserving the index
        var_list=df[cols].values.T # nicer code
        return f(var_list)

# Wasde
if True:
    def wasde_price_single(ticker, wasde_reports_series):
        df=read_security(ticker)
        df=df.resample('1d').ffill() # this to make sure that if there are holes, they will be filled

        # equivalent of the above line
        # new_index=pd.date_range(df.index.min(),df.index.max())
        # df=df.reindex(index=new_index,method='ffill')

        df = pd.merge(left=df, right=wasde_reports_series, left_index=True,right_index=True,how='left')
        df['report']=df['report'].fillna(method='ffill')
        df = df[['close_price','report']].groupby('report').mean()
        df=df.rename(columns={'close_price':ticker})

        return df    
    def parallel_wasde_price_single(single_variables, wasde_reports_series, parallel=None,max_workers=None):
        fo={}
        if parallel is None:
            for ticker in single_variables:
                fo[ticker] = wasde_price_single(ticker, wasde_reports_series)

        elif parallel=='thread':
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for ticker in single_variables:
                    results[ticker] = executor.submit(wasde_price_single, ticker, wasde_reports_series)
            
            for key, res in results.items():
                fo[key]=res.result()

        elif parallel=='process':
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for ticker in single_variables:
                    results[ticker] = executor.submit(wasde_price_single, ticker, wasde_reports_series)
            
            for key, res in results.items():
                fo[key]=res.result()

        # fo=pd.concat(fo,axis=1)
        # fo.columns = fo.columns.droplevel(level=1)

        fo = list(fo.values())
        fo=pd.concat(fo,axis=1)
        fo=fo.fillna(method='ffill')
        print('df_price_single.shape',fo.shape)
        return fo

    def wasde_price_multi(setting, wasde_reports_series, futures_calendar,  sel_years = range(1995,2024)):
        """
        example of 'setting':
            - setting={'security':'k1', 'start_month':5, 'prefix':'k_price_'}
            - setting={'security':None, 'start_month':None, 'prefix':'a_price_'}
        """
        fo={}
        ticker=setting['ticker']
        new_col_name=ticker+setting['suffix']

        if setting['delivery'] is not None:
            user_delivery=ticker+setting['delivery']
        else:
            user_delivery=None

        report_security_selection=wasde_report_security(ticker=ticker, futures_calendar=futures_calendar, wasde_report_dates=wasde_reports_series, user_delivery=user_delivery, user_start_month=setting['start_month'])
        
        sel_months=set([r[-2:-1] for r in report_security_selection])

        # Reading the price data according to 'sel_years' and 'sel_months'
        dfs={}
        for y in sel_years:
            for m in sel_months:
                sec = ticker+m+'_'+str(y)
                df=read_security(sec)
                df=df.resample('1d').ffill() # this to make sure that if there are holes, they will be filled
                df = pd.merge(left=df, right=wasde_reports_series, left_index=True,right_index=True,how='left')
                df['report']=df['report'].fillna(method='ffill')
                df = df[['close_price','report']].groupby('report').mean()
                df=df.rename(columns={'close_price':sec})            
                dfs[sec]=df

        df_price=pd.concat(dfs,axis=1)
        df_price.columns = df_price.columns.droplevel(level=0) # 'level' which level to drop (0: drops the first, 1: drop the second, etc etc)
            
        # Select according to 'report_security_selection'
        df_price=pd.melt(df_price, ignore_index=False)

        df_price=df_price.dropna(how='any') # this drops rows like: 'average price of 'w k_1998' for the report of 'may 2014'
        df_price['report_security']=[str(index.month)+"-"+relative_security( row['variable'],index.year) for index,row in df_price.iterrows()]

        mask = np.isin(df_price['report_security'],report_security_selection)
        df_price = df_price[mask]
        
        df_price = df_price.rename(columns={'variable':'security_'+new_col_name,'value':new_col_name})
        # df_price = df_price.drop(columns=['report_security','security_'+new_col_name])
        df_price = df_price.drop(columns=['report_security'])
        
        print(ticker)
        fo[new_col_name]=df_price
        return fo
    def parallel_wasde_price_multi(multi_variables, wasde_reports_series, futures_calendar,  sel_years = range(1995,2024), parallel=None, max_workers=None):
        """
        example of 'multi_variables':
            - multi_variables={'ticker':'c ', 'delivery':'k1', 'start_month':5, 'prefix':'k_price_'}
            - multi_variables={'ticker':'ng','delivery':None, 'start_month':None, 'prefix':'a_price_'}
        """
        fo={}
        if parallel is None:
            for setting in multi_variables:
                key=setting['ticker']+setting['suffix']
                fo[key] = wasde_price_multi(setting, wasde_reports_series, futures_calendar,  sel_years)

        elif parallel=='thread':
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for setting in multi_variables:
                    key=setting['ticker']+setting['suffix']
                    results[key] = executor.submit(wasde_price_multi, setting, wasde_reports_series, futures_calendar,  sel_years)
            
            for key, res in results.items():
                fo[key]=res.result()

        elif parallel=='process':
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results={}
                for setting in multi_variables:
                    key=setting['ticker']+setting['suffix']
                    results[key] = executor.submit(wasde_price_multi, setting, wasde_reports_series, futures_calendar,  sel_years)
            
            for key, res in results.items():
                fo[key]=res.result()

        fo = list(fo.values())
        fo =[list(df.values())[0] for df in fo]
        fo=pd.concat(fo, axis=1)
        fo=fo.fillna(method='ffill')
        print('df_price_multi.shape',fo.shape)
        return fo

    def wasde_report_security(ticker, futures_calendar=None,wasde_report_dates=None, user_delivery=None, user_start_month=None, return_full_df=False):
        """
        hello world
            - up.wasde_report_security('c ', futures_calendar, wasde_reports_series)

        it gives which future to use for a certain WASDE Report month (based on a futures calendar schedule)
        the output looks like this:

        report_month
        1      1-c h0
        2      2-c h0
        3      3-c k0
        4      4-c k0
        5      5-c n0
        6      6-c n0
        7      7-c z0
        8      8-c z0
        9      9-c z0
        10    10-c z0
        11    11-c z0
        12    12-c h1

        if I don't want the automatic calculation, it is necessary to provide:
            - user_delivery (like 'c k1'), to force the use of a specific security
            - user_start_month (like ), to force the use of a specific security

        return_full_df = True:
            - returns the table before selecting the 'most frequenct'
        """

        # Automatic Security Selection
        if user_delivery is None:
            fo={'report':[],'security':[]}

            if ('ticker' not in futures_calendar.columns):
                futures_calendar['ticker']=[info_ticker(sec) for sec in futures_calendar.index]

            for report_day in wasde_report_dates: 
                mask = ((futures_calendar['start']<=report_day) & (report_day<=futures_calendar['end']) & (futures_calendar['ticker']==ticker))
                fo['report'].append(report_day)
                fo['security'].append(futures_calendar[mask].index[0])    

            df=pd.DataFrame(fo)
            df['delivery']=[info_maturity(sec) for sec in df['security']]
            df['sec_month']=df['delivery'].dt.month
            df['report_month']=df['report'].dt.month
            df['report_year']=df['report'].dt.year
            
            df['relative_sec']=[relative_security(row['security'],row['report_year']) for index,row in df.iterrows()]    

            df=df.pivot(index='report_year',columns='report_month',values='relative_sec')

            if return_full_df:
                return df.sort_index(ascending=False)

            df=df.mode().T
            df['selection']=df.index.astype(str) +'-'+df[0]
            report_security_selection=df['selection']

        # User Specified
        else:        
            prev_offset=int(user_delivery[-1])-1
            prev_user_delivery = user_delivery[0:-1]+str(prev_offset)
            
            series_dict={}
            for m in range(1,user_start_month):
                series_dict[m]=str(m)+'-'+prev_user_delivery

            for m in range(user_start_month,13):
                series_dict[m]=str(m)+'-'+user_delivery

            report_security_selection=pd.Series(series_dict)

        return report_security_selection    