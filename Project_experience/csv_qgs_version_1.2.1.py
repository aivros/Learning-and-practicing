import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import math
import os   
import sys
import random
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

'''Code result : csv files query.
   Program description:                  
   1.Read csv: Read the CSV file from the current path.Generate a Data Frame dictionary.
   2.Correction dtypes: Determine the appropriate data frame column type.
   3.Row index dataframe: Generate a dataset dictionary on the column index. NOTE:df_dataframe_csv_index is main dataset dictionary. index_csv_main_dict is main index dictionary. 
   4.Filter: Row and column filtering. The parameters come from(7) the interaction function.(7.User-interface ===> 6.Dispatch hub===> 4.Filter)
   5.Statistical processing and output:  Statistical processing and visual output. 
   6.Dispatch hub: Accept the parameters of the interaction(7) function and call the filter function(4) to get the target data set, call the statistics and output functions(5). 
   7.User-interface: User query interaction based on the dictionary.(index_csv_main_dict)
   8.Main function: Program main flow control.
'''
#------------------------------------------------------------------------------------------------------------------------
# 1.read and write csv.
'''Contains three functions.Execute sequentially.If all is empty, the program is terminated in the main function.
'''
def csv_data_read():
    '''
    Generate a csv dictionary.csv_data_read is used to read the csv file and generate the dictionary CSV_DATA by the file name key.
    '''
    current_path = os.getcwd()
    files_list = os.listdir(current_path)
    CSV_DATA = {}    
    for i in files_list:                            
        if os.path.splitext(i)[1] == ".csv":  
            CSV_DATA[os.path.splitext(i)[0].lower().replace(' ', '_')] = i            
    return CSV_DATA

CSV_DATA = csv_data_read()
CSV_DATA_test = CSV_DATA #main function judging null values.

def csv_test_read(): 
    '''
    Exclude invalid CSV files.
    '''    
    CSV_DATA_Empty_list = []
    csv_data_keys_list = [b for b in CSV_DATA.keys()]
    for i in csv_data_keys_list:
        try:
            df_dataframe_csv_test = pd.read_csv(CSV_DATA[i])
        except:        
            CSV_DATA_Empty_list.append(CSV_DATA.pop(i))            
    return CSV_DATA,CSV_DATA_Empty_list

CSV_DATA,CSV_DATA_Empty_list = csv_test_read()
index_csv_list = [i.lower().replace(' ', '_') for i in CSV_DATA.keys()]
CSV_DATA_test_read = CSV_DATA #main function judging null values.

def csv_daraframe_read():
    '''
    Convert to a DataFrame based on different separators.
    '''
    df_dataframe_csv = {o.lower():[] for o in CSV_DATA.keys()} #Generate a empty dataframe dictionary.
    for index in df_dataframe_csv:
        try:
            df_dataframe_csv[index] = pd.read_csv(CSV_DATA[index])
            df_dataframe_csv[index].columns = [c.replace(' ', '_') for c in df_dataframe_csv[index].columns]
        except:
            try:
                df_dataframe_csv[index] = pd.read_csv(CSV_DATA[index],sep = ';')
                df_dataframe_csv[index].columns = df_dataframe_csv[index].columns.str.replace(' ', '_')
            except:
                print(CSV_DATA[index] + ' Separator error.')
    return df_dataframe_csv

df_dataframe_csv = csv_daraframe_read()#Used in the second part:2.Correction dtypes.


def dataframe_to_csv():
    current_path = os.getcwd()
    files_list = os.listdir(current_path)
          
    return CSV_DATA

#-----------------------------------------------------------------------------------------------------------------------------------
# 2.Correction dtypes.
'''Mainly includes three functions.csv_daraframe_dtype() is entrance.
'''
def csv_dataframe_Pretreatment():
    '''
    Compress the number of data rows.
    '''
    for y in df_dataframe_csv:
        if df_dataframe_csv[y].shape[0] >= 100000: #If the processing speed is too slow, resampling can be performed.
            sample_size = 100000
            sampled_row = pd.Series(df_dataframe_csv[y].index).sample(n = sample_size)
            df_dataframe_csv[y] = df_dataframe_csv[y].loc[sampled_row, :]
            df_dataframe_csv[y] = df_dataframe_csv[y].reset_index(drop = True)
    return df_dataframe_csv

df_dataframe_csv = csv_dataframe_Pretreatment()#Reduced sample data set.

time_system = r'\d{4}\-\d{2}\-\d{2}\s\d{2}\:\d{2}\:\d{2}'
time_wrold = r'\d{4}\/\d{2}\/\d{2}\s\d{2}\:\d{2}\:\d{2}'
time_ios8601 = r'\d{4}\.\d{2}\.\d{2}\s\d{2}\:\d{2}\:\d{2}'
time_utc = r'\d{4}\s\d{2}\s\d{2}\s\d{2}\:\d{2}\:\d{2}'
time_us = r'\d{4}\-\d{2}\-\d{2}'
time_gmt = r'\d{4}\/\d{2}\/\d{2}'
time_cet = r'\d{4}\.\d{2}\.\d{2}'
time_est = r'\d{4}\s\d{2}\s\d{2}'

data_time = {'a':time_system,
             'b':time_wrold,
             'c':time_ios8601,
             'd':time_utc,
             'e':time_us,
             'f':time_gmt,
             'g':time_cet,
             'h':time_est
            } #Time format regular expression.

dataframe_type = {'time':'datetime64[ns]','nan':'float64', 'int':'int64', 'float':'float64',
                  'bool':'bool', 'complex':'object', 'str':'object'}

def type_detection(df_dataframe_csv,y,e):
    python_type = {'time':[0],'nan':[0,None],'int':[0,1],'float':[0,0.1],  #Statistics and counting.
                   'bool':[0,True], 'complex':[0,123J] ,'str':[0,'a']}
    for m in df_dataframe_csv[y][e]:  #Traverse the dataframe column element type and count the number of similar python_type of the element.
        if (re.match(data_time['a'], str(m)) or re.match(data_time['b'], str(m)) or re.match(data_time['c'], str(m))
            or re.match(data_time['d'], str(m)) or re.match(data_time['e'], str(m)) or re.match(data_time['f'], str(m))
            or re.match(data_time['g'], str(m)) or re.match(data_time['h'], str(m))):
            python_type['time'][0] +=1
        elif pd.isnull(m):
            python_type['nan'][0] +=1
        elif type(python_type['int'][1]) == type(m):
            python_type['int'][0] +=1
        elif type(python_type['float'][1]) == type(m):
            python_type['float'][0] +=1
        elif type(python_type['bool'][1]) == type(m):
            python_type['bool'][0] +=1
        elif type(python_type['complex'][1]) == type(m):
            python_type['complex'][0] +=1
        else:
            python_type['str'][0] +=1

    comparison = {w: python_type[w][0] for w in python_type.keys()} #All types and quantities in this column.
    column_type = max(comparison.items(), key= lambda x: x[1])[0] #The one at most numerous types.'''
    try:
        if column_type == ('int' or 'float' or 'complex'):
            df_dataframe_csv[y][e] = pd.to_numeric(df_dataframe_csv[y][e], errors='coerce')
    except:
        if column_type == 'time':
            df_dataframe_csv[y][e] = pd.to_datetime(df_dataframe_csv[y][e], errors='coerce')
    return
  
def daraframe_cloumn_Branch(df_dataframe_csv,y):
    '''
    Exclude the canonical data column and focus on the object dtypes.
    '''
    for e in df_dataframe_csv[y].columns:   
        dtype_object_a =  pd.Series(['a'], dtype='object')
        
        if df_dataframe_csv[y][e].dtype == dtype_object_a.dtype:
            try:
                df_dataframe_csv[y][e] = pd.to_datetime(df_dataframe_csv[y][e])
            except:
                try:
                    df_dataframe_csv[y][e] = pd.to_timedelta(df_dataframe_csv[y][e])
                except:
                    try:
                        df_dataframe_csv[y][e] = pd.to_numeric(df_dataframe_csv[y][e])
                    except:
                        type_detection(df_dataframe_csv,y,e) 
        else:
            None
    return         
   
def csv_daraframe_dtype():
    '''
    'Correction dtypes' entrance.
    '''
    for y in df_dataframe_csv:
        daraframe_cloumn_Branch(df_dataframe_csv,y)        
    return df_dataframe_csv

df_dataframe_csv = csv_daraframe_dtype()

#--------------------------------------------------------------------------------------------------------------------
# 3.Row index dataframe.
'''Generate a dataset dictionary on the column index.
'''
def xxx(xxx): 
    '''
    The parameter of the map function is used as a condition for generating a new index column.object.
    '''
    xxx = str(xxx)
    i =  xxx[0]
    return i

def dataframe_index():
    '''
    dataframe_index is used to create indexed columns of different column types.
    '''
    index_csv_dict = {o.lower(): df_dataframe_csv[o].columns.tolist() for o in df_dataframe_csv.keys()}
    index_csv_main_dict = {} #The dataset dictionary that this query will use.
    for c_key in index_csv_dict:  #Based on the original data column, the loop generates a new index column.
        index_csv_main_dict[c_key] = {}
        for tm in index_csv_dict[c_key]:
            index_csv_main_dict[c_key][tm] = {}
            if df_dataframe_csv[c_key][tm].dtypes == 'datetime64[ns]':                
                df_dataframe_csv[c_key][tm + '_year'] = df_dataframe_csv[c_key][tm].dt.year.astype('category')
                index_csv_main_dict[c_key][tm][tm + '_year'] =list(set(list(df_dataframe_csv[c_key][tm + '_year'].dropna())))
                df_dataframe_csv[c_key][tm + '_month'] = df_dataframe_csv[c_key][tm].dt.month.astype('category')
                index_csv_main_dict[c_key][tm][tm + '_month'] = list(set(list(df_dataframe_csv[c_key][tm + '_month'].dropna())))                
                df_dataframe_csv[c_key][tm + '_day'] = df_dataframe_csv[c_key][tm].dt.day.astype('category')
                index_csv_main_dict[c_key][tm][tm + '_day'] = list(set(list(df_dataframe_csv[c_key][tm + '_day'].dropna())))
                df_dataframe_csv[c_key][tm + '_hour'] = df_dataframe_csv[c_key][tm].dt.hour.astype('category')
                index_csv_main_dict[c_key][tm][tm + '_hour'] = list(set(list(df_dataframe_csv[c_key][tm + '_hour'].dropna())))
                df_dataframe_csv[c_key][tm + '_week'] = df_dataframe_csv[c_key][tm].dt.weekday_name.astype('category')
                index_csv_main_dict[c_key][tm][tm + '_week'] = list(set(list(df_dataframe_csv[c_key][tm + '_week'].dropna())))
            elif df_dataframe_csv[c_key][tm].dtypes == 'object':
                df_dataframe_csv[c_key][tm + '_object'] = df_dataframe_csv[c_key][tm].map(xxx).astype('category')#Initial index.
                index_csv_main_dict[c_key][tm][tm + '_object'] = list(set(list(df_dataframe_csv[c_key][tm + '_object'].dropna())))
            elif df_dataframe_csv[c_key][tm].dtypes == 'bool':
                index_csv_main_dict[c_key][tm][tm] = list(set(list(df_dataframe_csv[c_key][tm].dropna()))) 

            else:
                quartile_index = df_dataframe_csv[c_key][tm].describe()
                quartile_category = [quartile_index['min']-1,quartile_index['25%'],quartile_index['50%'],quartile_index['75%'],quartile_index['max']+1]
                quartile_labels=['25%','50%','75%','100%']
                for _ in range(len(quartile_category)-1):
                    if quartile_category[_] == quartile_category[_+1]:
                        del quartile_labels[_]
                    else:
                        None
                if df_dataframe_csv[c_key][tm].dtypes == 'int64':
                    df_dataframe_csv[c_key][tm + '_int64'] = pd.cut(df_dataframe_csv[c_key][tm],quartile_category,labels=quartile_labels,duplicates='drop')
                    index_csv_main_dict[c_key][tm][tm + '_int64'] = quartile_labels
                elif df_dataframe_csv[c_key][tm].dtypes == 'float64':
                    df_dataframe_csv[c_key][tm + '_float64'] = pd.cut(df_dataframe_csv[c_key][tm],quartile_category,labels=quartile_labels,duplicates='drop')
                    index_csv_main_dict[c_key][tm][tm + '_float64'] = quartile_labels   

    return (df_dataframe_csv,index_csv_main_dict)

df_dataframe_csv_index,index_csv_main_dict = dataframe_index()#It's core data and dict.
#--------------------------------------------------------------------------------------------------------------------
# 4.Filter: Row and column filtering. 
'''Accept the parameters from the in_out_control_founction()(6.Dispatch hub) and return a data set.
'''
def filter_column(plot_data,column_filter,goal):  
    '''
    Parameters:
    plot_data : DataFrame.from filter_row returns.
    column_filter : list.equal parameters_column from csv_interactive_entry().
    goal : str
    Returns: DataFrame
    '''    
    filter_row_dataframe = plot_data       
    if column_filter == []:
        filter_column_dataframe = filter_row_dataframe[[goal]]
        return filter_column_dataframe
    column_group = [v for v in column_filter]
    column_group.append(goal)
    filter_column_dataframe = filter_row_dataframe[column_group]    
    return filter_column_dataframe

def filter_row(csv_main,cond):
    '''
    csv_main : csv_b is used to select the df_dataframe_csv_index.
    cond : cond is an index keyword filtering dictionary.
    Returns: DataFrame
    '''
    filter_row_data = df_dataframe_csv_index[csv_main]
    if cond == {}:
        return filter_row_data       
    for w in cond.keys():        
        for v in cond[w].keys():
            or_bool = [] 
            for c in range(len(cond[w][v])):
                t = str(c)
                t = filter_row_data[v] == cond[w][v][c]
                or_bool.append(t)
            if len(or_bool) > 1:
                for n in range(1,len(or_bool)):
                    or_bool[0] = or_bool[0] | or_bool[n]
                filter_row_data = filter_row_data[or_bool[0]].reset_index(drop = True)
            else:
                filter_row_data = filter_row_data[or_bool[0]].reset_index(drop = True)
               
    return filter_row_data
#---------------------------------------------------------------------------------------------------------------------------
# 5.Statistical processing and output.
'''Accept the parameters from the in_out_control_founction()(6.Dispatch hub) and output descriptive statistics.
'''
def color_x(): 
    '''
    Artist's histogram graffiti.for plot color.
    '''
    color = np.random.rand(1,3)
    return color


def out_print_count(csv_main,parameters_column,condition,goal,plot_data): 
    '''Output not numerical statistical results.
    '''
    if parameters_column == []:
        if  plot_data.shape[0]>= 10:
            popular_list_a = plot_data[goal].value_counts()
            popular_list = popular_list_a[:10]
        else:
            popular_list_a = plot_data.value_counts()
            popular_list = popular_list_a[:plot_data.shape[0]]
        plot_data_p = pd.Series(popular_list)
        only_count = len(popular_list)
        plt.figure(figsize=(15,8))
        show = plot_data_p.plot(kind = 'bar', color = color_x())
    else:        
        plt.figure(figsize=(15,8))
        show = plot_data.groupby(parameters_column)[goal].value_counts(normalize = False,sort = False).sort_values(ascending=False).iloc[:31].plot(kind = 'bar',logy = True, color = color_x())#iloc[:31]:Limit the output range to avoid rendering stuck.
        only_count = '<= 31'
    oo,labels = plt.xticks()
    show.set_xticklabels(labels,rotation=30,ha="right");
    parameters_column = [p.replace('_', '-') for p in parameters_column]
    xaxis_legend_a = '-'.join(parameters_column)
    plt.ylabel('Counts_number')
    plt.title('{} ingredients'.format(goal), fontsize =18)
    plt.show()
    xaxis_legend = []  #Generate an input condition string.
    for g in condition.keys():
        xaxis_legend.append(str(g))
        if type(condition[g]) != type([' ']):
            for s in condition[g].keys():
                xaxis_legend.append(str(s))
                for l in condition[g][s]:
                    xaxis_legend.append(str(l))
        else:
            for e in condition[g]:
                xaxis_legend.append(str(e))
                xaxis_legend = [x.replace('_', '-') for x in xaxis_legend]
    xaxis_legend_o = '-'.join(xaxis_legend)
    data_count = df_dataframe_csv_index[csv_main].shape[0]
    this_count = plot_data.shape[0]    
    print('This is a frequency statistics display of "{}" in the "{}" range of "{}".'.format(goal,xaxis_legend_o,csv_main))
    print('There are {:d} readings ({:.2f}%) matching the filter criteria.'.format(this_count, 100. * this_count / data_count))
    print('This is the highest frequency of the first {} elements.'.format(only_count))
    #print()
    return

def out_print_value(csv_main,parameters_column,condition,goal,plot_data):
    '''Output numerical statistical results.
    '''
    if parameters_column == []:
        plt.figure(figsize=(15,8))
        d = {'mean' : plot_data[goal].mean() , goal : goal}
        index = [0]
        df = pd.DataFrame(data=d, index=index)
        show = df.plot(x= goal, y= 'mean', kind = 'bar', color = color_x())
    else:        
        plt.figure(figsize=(15,8))
        show = plot_data.groupby(parameters_column)[goal].mean().plot(kind = 'bar', color = color_x())
    oo,labels = plt.xticks()
    show.set_xticklabels(labels,rotation=30,ha="right");
    ylavel_print = goal
    plt.ylabel(ylavel_print)
    plt.title('{} ingredients'.format(goal), fontsize =18)
    plt.savefig('tmp.pdf', bbox_inches='tight')
    plt.show()
    xaxis_legend = []  #Generate an input condition string.
    for g in condition.keys():
        xaxis_legend.append(str(g))
        if type(condition[g]) != type([' ']):
            for s in condition[g].keys():
                xaxis_legend.append(str(s))
                for l in condition[g][s]:
                    xaxis_legend.append(str(l))
        else:
            for e in condition[g]:
                xaxis_legend.append(str(e))
                xaxis_legend = [x.replace('_', '-') for x in xaxis_legend]
    xaxis_legend_o = '-'.join(xaxis_legend)
    data_count = df_dataframe_csv_index[csv_main].shape[0]
    this_count = plot_data.shape[0]
    print('This is a frequency statistics display of "{}" in the "{}" range of "{}".'.format(goal,xaxis_legend_o,csv_main))
    print('There are {:d} readings ({:.2f}%) matching the filter criteria.'.format(this_count, 100. * this_count / data_count))

    return


#-------------------------------------------------------------------------------------------------------------------------------------
# 6.Dispatch hub.
''' Accept the parameters from the interaction function, filter the rows first, and then filter the columns from the results.
    Give the target data set to the output function.  
'''
def in_out_control_founction(csv_main,parameters_column,condition,goal):
    '''
    Parameters: from csv_interactive_entry() of (7)User-interface.
    csv_main : string
    Used for selection of different CSV data sets.Is the key of df_dataframe_csv_index.
    parameters_column : list
    Used for the selection of related columns.
    condition : dict
    Used for the selection of row.
    goal : string
    a column name of df_dataframe_csv_index[csv_main].only one.
    '''
    plot_data = filter_row(csv_main,condition)#Call the filter.
    if plot_data.empty:   #Exclude the situation for there are no data that match the filter.
        print('There are no eligible results.')
        return
    else:
        plot_data = filter_column(plot_data,parameters_column,goal)#Call the filter.
    if plot_data[goal].dtype == 'int64' or plot_data[goal].dtype == 'float64':
        out_print_value(csv_main,parameters_column,condition,goal,plot_data)#processing and output for numerical type.
    else:
        out_print_count(csv_main,parameters_column,condition,goal,plot_data)#processing and output for not numerical type.
    return 

#-------------------------------------------------------------------------------------------------------------------------------------
# 7.User-interface.
'''entrance: csv_interactive_entry(). 
   According to the result of the interaction, get the parameters and call the function:in_out_control_founction(main_csv,parameters_column,condition,goal);      
'''       
def main_csv_select(index_cav_list):
    '''
    main_csv_select:Target CSV returns parameters.
    '''    
    while True: 
        print(index_cav_list)
        message_n = "\nPlease choose one of them.\n"
        print(message_n)
        message_input_n = "\nPlease copy one from the list and enter: \n"
        input_n = input(message_input_n).lower()
        if input_n in index_cav_list:
            main_csv = input_n
            return main_csv
        else:
            print('\nThe content you entered is not in the scope of this inquiry! Please enter again!\n')
    return     
                           
def goal_select(main_csv):
    guide_mode_b = "\nThe target is the feature to be queried. (Note: only one can be selected!)\n"
    print(guide_mode_b)
    message_g = '\nplease selecte query goal:\n'
    print(message_g)
    column_list = [i for i in df_dataframe_csv_index[main_csv].columns]
    print(column_list)
    while True:
        message_input_a = "\nPlease copy one from the list and enter:\n"
        input_a = input(message_input_a)
        if input_a in column_list:
            goal = input_a
            return goal
        else:
            print('\nThe content you entered is not in the scope of this inquiry! Please enter again!\n')
    return
    
def columns_select(main_csv,goal):
    guide_columns_select = "\nPlease select columns in the above list:\n"
    print(guide_columns_select)
    column_list = [str(i) + '.' + str(v) for i, v in enumerate(df_dataframe_csv_index[main_csv].columns)]
    column_list = [i for i in df_dataframe_csv_index[main_csv].columns if i != goal]
    column_list.append('none')  
    return_parameters_column = []    
    while True:
        print(column_list)
        message_h = "\nPlease copy one of them.\n"
        input_h = input(message_h) 
        if input_h == 'none':
            return_parameters_column = []
            return return_parameters_column
        if input_h in column_list and input_h not in return_parameters_column:
            return_parameters_column.append(input_h)
            message_again = '\nStill have to choose?(y/n)\n'
            input_m_a = input(message_again).lower()
            if input_m_a != 'y':
                print(return_parameters_column)
                return return_parameters_column                                
        else:
            print('\nThe content you entered is not in the scope of this inquiry! Please enter again!\n')    
    return 
    
def condition_select(main_csv,goal):                           
    message_c = '\nplease selecte query condition:\n'
    print(message_c)
    Primary_directory = [i for i in index_csv_main_dict[main_csv].keys() if i != goal]
    Primary_directory.append('all')
    condition_dict = {}
    while True:
        print(Primary_directory)        
        message_input_b = "\nPlease copy one from the primary directory and enter:\n"
        input_b = input(message_input_b)
        if input_b =='all':
            condition_dict = {}
            return condition_dict
        if input_b in Primary_directory and input_b not in condition_dict.keys():
            condition_dict = primary_secondary(main_csv,goal,condition_dict,input_b)            
            message_end = '\nDoes it continue condition?(y/n)\n'
            input_e = input(message_end).lower()
            if input_e != 'y':
                return condition_dict               
        else:
            print('\nThe content you entered is not in the scope of this inquiry! Please enter again!\n') 
    return  

def primary_secondary(main_csv,goal,condition_dict,input_b):
    condition_dict[input_b] = {}
    secondary_directory = [i for i in index_csv_main_dict[main_csv][input_b].keys() if i != goal]
    while True: 
        print(secondary_directory)
        message_input_c = "\nPlease copy one from the secondary directory and enter:\n"
        input_c = input(message_input_c)
        if input_c in secondary_directory:
            condition_dict = secondary_directory_list(main_csv,condition_dict,input_b,input_c)
            return condition_dict
        else:
            print('\nThe content you entered is not in the scope of this inquiry! Please enter again!\n')
    return                
                    
def secondary_directory_list(main_csv,condition_dict,input_b,input_c):
    condition_dict[input_b][input_c] = []
    print(input_b,input_c)
    while True: 
        print(index_csv_main_dict[main_csv][input_b][input_c])
        select_dict = {str(i).lower(): i for i in index_csv_main_dict[main_csv][input_b][input_c]}
        message_input_d = "\nPlease copy one from the secondary directory list and enter:\n"
        input_d =  input(message_input_d).lower()
        if input_d in select_dict.keys():
            if select_dict[input_d] in index_csv_main_dict[main_csv][input_b][input_c] and input_d not in condition_dict[input_b][input_c]:
                condition_dict[input_b][input_c].append(select_dict[input_d])
                message_end = '\nDoes it continue?(y/n)\n'
                input_e = input(message_end).lower()
                if input_e != 'y':
                    return condition_dict
    return 

def csv_interactive_entry():
    '''
    'User-interface' entrance.
    '''
    while True:
        message_fill = '|'.join(index_csv_list)
        message = "Hi ^_^, Welcome to CSV Query System (Version 1.2)!\nThis query is about the composition characteristics and quantity statistics of something whith:\n\n{}.\n\nEnjoy!\n".format(message_fill)                
        print(message)

        main_csv = main_csv_select(index_csv_list)  
        goal = goal_select(main_csv)
        parameters_column = columns_select(main_csv,goal)
        condition = condition_select(main_csv,goal)#===>primary_secondary()===>secondary_directory_list()

        in_out_control_founction(main_csv,parameters_column,condition,goal)

        input_restar = input("\nplease input 'y' to restar:\n").lower()
        if input_restar == 'y':
            print ('\nWelcome back!\n')
        else:
            break
    return   

#def quit_sometime():
    #quit = sys.stdin.read(1)
    #if quit == '\x1b':
        #os.exit()
        #return

#-------------------------------------------------------------------------------------------------------------------
# 8.Main function.
def main(): 
    try:        
        if CSV_DATA_test == {}:
            return print('There are no csv files.')
        if CSV_DATA_test_read == {}:
            return print('There are no csv files available.',CSV_DATA_Empty_list)
        csv_interactive_entry() 
    except:
        print('opps! I broke down. :( ')
        return
    return print ('Welcome again!')
if __name__ == "__main__":
    main()
