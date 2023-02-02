'''
https://docs.bokeh.org/en/test/docs/reference/models/widgets/inputs.html#bokeh.models.MultiSelect
https://docs.bokeh.org/en/2.4.2/docs/user_guide/interaction/callbacks.html
'''

# Preliminaries
if True:
    from datetime import datetime as dt
    import streamlit as st
    from bokeh.models import CustomJS, MultiSelect, AutocompleteInput
    from streamlit_bokeh_events import streamlit_bokeh_events
    import pandas as pd
    import warnings; warnings.filterwarnings("ignore")

    import Prices as up
    import Charts as uc
    import GDrive as gd
    import plotly.express as px

    special_cases={'caz':'cax'}
    st.set_page_config(page_title='Seasonals',layout='wide',initial_sidebar_state='expanded')

    service = gd.build_service()

    if 'cloud_map_dict' not in st.session_state:
        st.session_state['cloud_map_dict']=up.get_cloud_sec_map(service=service)
    if 'all_dates' not in st.session_state:
        st.session_state['all_dates']=up.read_dates(service=service)
    if 'seas_df' not in st.session_state:
        st.session_state['seas_df']=[]
    if 're_run' not in st.session_state:
        st.session_state['re_run']=True
    cloud_map_dict=st.session_state['cloud_map_dict']
    all_dates=st.session_state['all_dates']
    
    var_options=['close_price',
    'open_price',
    'low_price',
    'high_price',
    'open_interest',
    'volume',

    'implied_vol_dm',
    'implied_vol_dm_call_25d',
    'implied_vol_dm_put_25d',
    'implied_vol_dm_skew_25d',

    # 'implied_vol_hist_call',
    # 'implied_vol_call_50d',
    # 'implied_vol_hist_put',
    # 'implied_vol_put_50d',
    # 'bvol_50d',

    # 'implied_vol_call_25d',
    # 'bvol_call_25d',

    # 'implied_vol_put_25d',            
    # 'bvol_put_25d',
    ]

# Functions
if True:
    def format_timeframe_date(item):
        return item.strftime("%b %Y")
    
    def reverse_dict(original_dict):
        return {v: k for k, v in original_dict.items()}
    
    def symbols_special_cases(symbols, special_cases):
        '''
        ex:
            - cax: 1999 ->  2014
            - caz: 2014 ->  present
        so what I want to do is: everytime I want to analyze 'caz', I need to add 'cax' and treat it like it was 'caz' (it will be)
            
        special_cases={'caz':'cax'}
            - 

        '''
        fo={}
        for k,v in symbols.items():
            fo[k]=v

            if k in special_cases:
                new_key=up.dm_replace(k,special_cases)
                fo[new_key]=new_key.replace(' ','_')
        return fo

    def sec_dfs_special_cases(sec_dfs, special_cases):
        fo={}
        special_cases_r=reverse_dict(special_cases)
        for k,v in sec_dfs.items():
            fo[up.dm_replace(k,special_cases_r)]=v

        return fo

# Events
if True:
    def y_ticker_on_change():
        st.session_state['bokeh_multiselect_on_change']=[]
        st.session_state['seas_df']=[]
        st.session_state['re_run']=True
        st.session_state['y_expression']=''

    def y_expression_on_change():
        st.session_state['bokeh_multiselect_on_change']=[]
        st.session_state['seas_df']=[]
        st.session_state['re_run']=True
        st.session_state['y_ticker']=''

    def sec_selection_on_change():
        st.session_state['bokeh_multiselect_on_change']=[]
        st.session_state['seas_df']=[]
        st.session_state['re_run']=True

# Controls
if True:
    options=up.select_securities(include_continuous=False, cloud_map_dict=cloud_map_dict)
    options=list(set([up.info_ticker_and_letter(s) for s in options]))
    options=list(set(options)-set(special_cases.values()))
    options.sort()

    # col1, col2 = st.columns([1,1])
    tab1, tab2 = st.tabs(['Simple Ticker', 'Custom Expressions'])
    with tab1:
        col1, col2, col3, col4 = st.columns([4,1,0.5,0.5])
        with col1:
            ticker_selection = st.selectbox('Ticker',['']+options, key='y_ticker', on_change=y_ticker_on_change)            
        with col2:
            ticker_var = st.selectbox('Variable',var_options, var_options.index('close_price'),  key='ticker_var', on_change=sec_selection_on_change)
        with col3:
            ticker_range = st.number_input('Range (Months)',1,24,6,3,key='ticker_range', on_change=sec_selection_on_change)
        with col4:
            ticker_past_years = st.number_input('Past Years',1,100,10,5,key='ticker_years', on_change=sec_selection_on_change)

    with tab2:
        col1, col2, col3, col4 = st.columns([4,1,0.5,0.5])
        with col1:
            expression_selection = st.text_input("'July Nov':  s n - s x, 'Dec wheat-corn': w z - c z, 'Soy Corn Ratio': s x / c z, etc...", key='y_expression', on_change=y_expression_on_change)
        with col2:
            expression_var = st.selectbox('Variable',var_options, var_options.index('close_price'),  key='expression_var', on_change=sec_selection_on_change)
        with col3:
            expression_range = st.number_input('Range (Months)',1,24,6,3,key='expression_range', on_change=sec_selection_on_change)
        with col4:
            expression_past_years = st.number_input('Past Years',1,100,10,5,key='expression_years', on_change=sec_selection_on_change)

    expression=''
    if ticker_selection!='':
        expression=ticker_selection
        var_selection=ticker_var
        seas_range=ticker_range
        past_years=ticker_past_years

    elif expression_selection!='':
        expression=expression_selection
        var_selection=expression_var
        seas_range=expression_range
        past_years=expression_past_years
        
    if (expression == ''): st.stop()
    
    seas_interval=[dt.date(dt.today()-pd.DateOffset(months=6)+pd.DateOffset(days=1)), dt.date(dt.today()+pd.DateOffset(months=seas_range))]    

# Calculations
if True:   
    fnd=[] 
    seas_df = st.session_state['seas_df']

    # Core Calc
    if len(seas_df)==0:
        with st.spinner('Downloading Data...'):
            symbols=up.extract_symbols_from_expression(expression)                
            symbols = symbols_special_cases(symbols, special_cases)
            print('symbols',symbols)

            sel_sec=[]
            for s in symbols:
                sel_sec=sel_sec+up.select_securities(ticker_and_letter=up.info_ticker_and_letter(up.symbol_no_offset(s)), cloud_map_dict=cloud_map_dict)
            
            # sec_dfs = {'w n_2020' : df}            
            sec_dfs= up.read_security_list(sel_sec, parallel='thread')
            sec_dfs=sec_dfs_special_cases(sec_dfs, special_cases)

        with st.spinner('Making the Seasonals Calculation...'):
            ref_year=dt.today().year
            fnd=min([all_dates.loc[sec]['first_notice_date'] for sec in sel_sec if up.info_maturity(sec).year==ref_year])
            
            if '_skew_' in var_selection:
                for key, df in sec_dfs.items():
                    df['implied_vol_dm_skew_25d']=df['implied_vol_dm_call_25d']-df['implied_vol_dm_put_25d']

            sec_dfs=up.sec_dfs_simple_sec(sec_dfs)

            print('fnd:',fnd)

            st.session_state['seas_df']=up.create_seas_df(expression, sec_dfs, var_selection, ref_year=ref_year, seas_interval= [seas_interval[0], min(fnd, seas_interval[1])])
            seas_df=st.session_state['seas_df']
    
    if True:
        col1, col2, col3 = st.columns([12,0.5,1.5])
        # Years Selection
        with col3:
            seas_only = st.checkbox('Seas only')
            options = list(seas_df.columns)
            options.sort()
            options.reverse()
            options = ['mean'] + [f'{o}' for o in options]
            
            # Create the MultiSelect widget   
            pre_selection=options[0: min(past_years, len(options))+2]
            bokeh_multiselect = MultiSelect(value=pre_selection, options=options, size = 40, width =100)
            bokeh_multiselect.js_on_change("value", CustomJS(args=dict(xx='Hello Daniele'), code='console.log(xx.toString());document.dispatchEvent(new CustomEvent("GET_OPTIONS", {detail: this.value}));'))
            sel_years = streamlit_bokeh_events(bokeh_multiselect,events="GET_OPTIONS",key='bokeh_multiselect_on_change', override_height=750, debounce_time=200, refresh_on_update=False)

        if (sel_years is None) or len(sel_years)==0 or (st.session_state['re_run']):
            cols=[y for y in pre_selection]
        else:
            cols=[y for y in sel_years['GET_OPTIONS']]

        # Chart
        seas_cols=[int(x) if x.isnumeric() else x for x in cols]
        with col1:
            fig = uc.seas_chart(seas_df, seas_cols, seas_only)
            st.plotly_chart(fig,use_container_width=True, config={'scrollZoom': True, 'displayModeBar':False})

    # Re-Run hack
    if st.session_state['re_run']:
        st.session_state['re_run']=False        
        st.experimental_rerun()