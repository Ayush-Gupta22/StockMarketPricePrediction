import streamlit as st
import yfinance as yf  #Download market data from Yahoo! Finance API
                       #The API serves real-time and historical data for crypto and stock markers.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #Feature Scaling Technique used to standardize independent
# variables so that there is no dominance of one independent variable over other  x' = (x-mean(x))/standard deviation
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from plotly.subplots import make_subplots
from pandas_datareader import data
import datetime as dt
from datetime import timedelta 

st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)


@st.cache
def grid_search(model, x_train, y_train):
    grid_rf = {
    'n_estimators': [20, 50, 100, 500, 1000],  
    'max_depth': np.arange(1, 15, 1),  
    'min_samples_split': [2, 10, 9], 
    'min_samples_leaf': np.arange(1, 15, 2, dtype=int),  

    'bootstrap': [True, False], 
    'random_state': [1, 2, 30, 42]
    }
    grid = GridSearchCV(estimator = model,param_distributions=grid_rf, cv=3, n_jobs=-1, verbose=2, n_iter=200)
    grid_fit = grid.fit(x_train, y_train)
    best_parameters = grid_fit.best_params_
    return best_parameters
    
@st.cache
def stock_download(ticker):
    data1 = yf.download(ticker, start="2007-01-01", end="2022-05-19")#Downloading data
    return data1

def func(ticker):
    data1 = stock_download(ticker)
    
    with st.expander("Show Data"):
        st.dataframe(data1)
    df1 = pd.DataFrame(data1)  #Aligning data in row and column 
    df1.to_csv("data.csv")  #converts DataFrame into CSV data.
    st.markdown("##")
    
    read_df1 = pd.read_csv("data.csv")
    read_df1.set_index("Date", inplace=True)  #used to set the DataFrame index using existing columns.
                                        # inplace=True means change the same object do not make new object.
    #st.dataframe(read_df1)
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;">Graph of Open Price and Close Price vs Date </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    plt.figure(figsize=(9, 3))
    read_df1['Open'].plot(color = "blue")
    read_df1['Close'].plot(color = "yellow")
    plt.ylabel("Stock Prices")
    plt.legend()
    st.pyplot(clear_figure=True)
    st.markdown("##")
    st.markdown("##")
        
        
    
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;">Graph of High Price and Low Price vs Date </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    plt.figure(figsize=(9, 3))
    read_df1['High'].plot(color = "red")
    read_df1['Low'].plot(color = "green")
    plt.ylabel("Stock Prices")
    plt.legend()
    st.pyplot(clear_figure=True)
    st.markdown("##")
    st.markdown("##")
        
    
    data1 = pd.read_csv("data.csv") 
    data1.set_index("Date", inplace=True)   #used to set the DataFrame index using existing columns.
                                        # inplace=True means change the same object do not make new object.
    data1.dropna(inplace=True)   #Remove Missing Value or Drop Rows/Columns with missing value

    x = data1.iloc[:, 0:5].values
    y = data1.iloc[:, 4].values
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.26, random_state=0)
    #Random state is defined so that train data will be constant For every run so that it will make easy to debug.
    
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #st.dataframe(x_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, min_samples_split=6, min_samples_leaf=2, max_depth=15, bootstrap=True)
    model.fit(x_train, y_train)
    
    
    st.markdown("##")
    predict = model.predict(x_test)
    
    analysis = '<h2 style="font-family:sans serif; color:#007fff; font-size: 40px; font-style: italic;">Analysis</h3>'
    st.markdown(analysis, unsafe_allow_html=True)
    error = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;">Errors</h3>'
    st.markdown(error, unsafe_allow_html=True)
    st.write("1. Mean Absolute Error:", round(metrics.mean_absolute_error(y_test, predict), 4))
    st.write("2. Mean Squared Error:", round(metrics.mean_squared_error(y_test, predict), 4))
    st.write("3. Root Mean Squared Error:", round(np.sqrt(metrics.mean_squared_error(y_test, predict)), 4))
    st.markdown("##")
    
    st.markdown("##")
    pre = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;">Prediction for next 7 days</h3>'
    st.markdown(pre, unsafe_allow_html=True)
    predictions = pd.DataFrame({"Date": pd.date_range(start=data1.index[-1], periods=len(predict)), "Predictions of Close Price": predict}, index=pd.date_range(start=data1.index[-1], periods=len(predict)))
    # index = date from where the predict data will start. df.index[-1] = start from the last date present in df variable.
    # period = no of days you wants from starting date
    predictions = predictions.reset_index(drop = True)
    predictions['Date'] = pd.to_datetime(predictions['Date']).dt.date
    #predictions = predictions.reset_index()
    sevendays_df = pd.DataFrame(predictions[3:10])
    sevendays_df.to_csv("five-days-predictions.csv")
        
    sevendays_df_pred = pd.read_csv("five-days-predictions.csv", index_col=0)
    st.dataframe(sevendays_df_pred)
    #fivedays_df_pred.set_index("Date", inplace=True)
    #fivedays_df_pred = fivedays_df_pred.to_string(index = False)\
    st.markdown("##")
    buy_price = min(sevendays_df_pred["Predictions of Close Price"])
    sell_price = max(sevendays_df_pred["Predictions of Close Price"])
    sevendays_buy = sevendays_df_pred.loc[sevendays_df_pred["Predictions of Close Price"] == buy_price]
    sevendays_sell = sevendays_df_pred.loc[sevendays_df_pred["Predictions of Close Price"] == sell_price]
    day_buy = '<h4 style="font-family:sans serif; color:#F0dc82; font-size: 20px;">Date and Buy Price</h4>'
    st.markdown(day_buy, unsafe_allow_html=True)
    st.write(sevendays_buy)
    day_sell = '<h4 style="font-family:sans serif; color:#F0dc82; font-size: 20px;">Date and Sell Price</h4>'
    st.markdown(day_sell, unsafe_allow_html=True)
    st.write(sevendays_sell)
    day_graph = '<h4 style="font-family:sans serif; color:#F0dc82; font-size: 20px;">Graph: Predicted Close Price vs Date</h4>'
    st.markdown(day_graph, unsafe_allow_html=True)
    plt.figure(figsize=(10, 3))
    plt.title("Forecast for the next 7 days")
    plt.plot(sevendays_df_pred["Date"], sevendays_df_pred["Predictions of Close Price"], color="green")
    #fivedays_df_pred["Predictions of Close Price"].plot(figsize=(10, 5), title="Forecast for the next 5 days", color="green")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot() 

#creating dataset matrix according to timestamp
def create_dataset(dataset , timestamp=1):
    data_X , data_Y =[] , []
    for i in range(len(dataset)-timestamp-1):
        data_X.append(dataset[i:(i+timestamp),0])
        data_Y.append(dataset[timestamp+i:timestamp+i+1,0:1])
    return np.array(data_X) , np.array(data_Y)

def model(x_train, y_train, x_test , y_test):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Dropout
    from tensorflow.keras.layers import LSTM
    
    models = Sequential()
    models.add(LSTM(50,return_sequences=True, input_shape=(100,1)))
    models.add(Dropout(0.2))

    models.add(LSTM(50,return_sequences=True))
    models.add(Dropout(0.2))

    models.add(LSTM(50))
    models.add(Dropout(0.2))

    models.add(Dense(1))
    models.compile(loss='mean_squared_error', optimizer='adam')

    models.fit(x_train, y_train , validation_data=(x_test, y_test),epochs=15, batch_size=64 , verbose=1 )
    return models
    
def funcLSTM(ticker):
    dataLSTM = stock_download(ticker)
    df_LSTM = pd.DataFrame(dataLSTM)  #Aligning data in row and column 
    df_LSTM.to_csv("data1.csv")  #converts DataFrame into CSV data.
    
    read_df_LSTM = pd.read_csv("data1.csv")
    with st.expander("Show Data"):
        st.dataframe(read_df_LSTM)
    x = read_df_LSTM.filter(['Low'])
    x = x.to_numpy()
    
    dates_for_later_use = read_df_LSTM['Date']
    
    read_df_LSTM["Date"] = pd.to_datetime(read_df_LSTM["Date"])
    date = read_df_LSTM["Date"]
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;">Graph of Low Price vs Date </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 4))
    year_month_formatter = mdates.DateFormatter("%Y-%m")
    ax.xaxis.set_major_formatter(year_month_formatter)
    ax.plot(date, x)
    fig.autofmt_xdate()
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    st.pyplot()
    
    #Data Preprocessing
    #Scaling
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler(feature_range=(0,1))
    x=scaler.fit_transform(np.array(x).reshape(-1,1))
    
    #Test - Train Split
    train_data=x[0:int(len(x)*0.75),:]
    test_data=x[int(len(x)*0.75):int(len(x)),:]

    timestamp=100
    x_train , y_train=create_dataset(train_data, timestamp)
    x_test , y_test=create_dataset(test_data, timestamp)
    
    x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],1)
    x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],1)

    md = model(x_train, y_train, x_test , y_test)

    
    train_predict=md.predict(x_train)
    test_predict=md.predict(x_test)


    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)


    
    test_data_size = test_data.shape[0]
    #print(test_data_size)
    x_input=test_data[test_data_size-100:].reshape(1,-1)
    st.markdown("##")
    st.markdown("##")

    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()

    lst_output=[]
    n_steps=100
    i=0
    while(i<15):
        
        if(len(temp_input)>100):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = md.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = md.predict(x_input, verbose=0)
            #print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            #print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
        
    lst_output_act = scaler.inverse_transform(lst_output)
    
    array_for_output = [0]*30

    xx =read_df_LSTM.iloc[len(x)-15:,3:4].values
    
    dates2 = [0]*30
    for i in range(15) :
        array_for_output[i] = xx[i]
        array_for_output[i+15] = lst_output_act[i]
        dates2[i] = dates_for_later_use[len(x)-15+i]
        dates2[i+15] = date[len(x)-1]+timedelta(days = i+1)

    const = array_for_output[14]-array_for_output[15]
    for i in range(15,30):
        array_for_output[i] = array_for_output[i]+const
    
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;">Future Prediction of 15 days </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    dates2 = pd.to_datetime(dates2)
    labels = [1,14,30]
    xtiks = [dates_for_later_use[len(x)-15],dates2[15].date(),dates2[29].date()]  
    plt.xticks( labels,xtiks,rotation =45)
    plt.plot(array_for_output)
    plt.title('15 days earlier  +  15 days predicted stock prices')
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    st.pyplot()

def funcRegression(ticker):
    dataLR = stock_download(ticker)
    df_LR = pd.DataFrame(dataLR)  #Aligning data in row and column 
    df_LR.to_csv("data2.csv")  #converts DataFrame into CSV data.
    st.markdown("##")

    read_df_LR = pd.read_csv("data2.csv")
    with st.expander("Show Data"):
        st.dataframe(read_df_LR)
    
    read_df_LR['index_column'] = read_df_LR.index
    # df = pd.DataFrame(df, columns = ['Date','Open','High', 'Low' , 'Close' , 'Volume' ,  'OpenInt'])

    leng = len(read_df_LR)
    x = read_df_LR.iloc[:,7:8].values
    a = read_df_LR.iloc[:,2:3].values
    b = read_df_LR.iloc[:,3:4].values


    y = (a+b)/2


    # splitting into training and test set
    from sklearn.model_selection import train_test_split
    x_train , x_test , y_train , y_test=train_test_split(x,y, test_size=1/3, random_state=0)
    
    #fitting linear regression
    from sklearn.linear_model import LinearRegression
    reg=LinearRegression()
    reg.fit(x_train,y_train)


    matrix = []

    for r in range(0, 30):
        matrix.append([r+leng for c in range(0, 1)])

    #predicting
    y_pred=reg.predict(matrix)
    
    
    #visualizing training set result
    st.markdown("##")
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;"> Visualizing Training Dataset </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    plt.figure(figsize = (10, 3))
    plt.scatter(x_train, y_train, color='red')
    plt.plot(x_train, reg.predict(x_train) , color='blue')
    plt.title('Date vs Low Price')
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    st.pyplot()
    
    #visualizing test set result
    st.markdown("##")
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;"> Visualizing Testing Dataset </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    plt.figure(figsize=(10, 3))
    plt.scatter(x_test, y_test, color='red')
    plt.plot(x_train, reg.predict(x_train), color='blue')
    plt.title('Date vs Low Price')
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    st.pyplot()

    #visualizing forcasting data set
    st.markdown("##")
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;"> Visualizing Forcasting Value </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    plt.figure(figsize = (10, 3))
    plt.plot(matrix, y_pred, color='blue')
    plt.title('Date vs Low Price')
    plt.xlabel('Date')
    plt.ylabel('Low Price')
    st.pyplot()

def funcPoly(ticker):
    dataPoly = stock_download(ticker)
    df_Poly = pd.DataFrame(dataPoly)  #Aligning data in row and column 
    df_Poly.to_csv("data3.csv")  #converts DataFrame into CSV data.
    
    df= pd.read_csv('data3.csv')
    with st.expander("Show Data"):
        st.dataframe(df)
    df['index_column'] = df.index
    # df = pd.DataFrame(df, columns = ['Date','Open','High', 'Low' , 'Close' , 'Volume' ,  'OpenInt'])


    leng=len(df)

    x = df.iloc[leng-90:,7:8].values
    a = df.iloc[leng-90:,2:3].values
    b = df.iloc[leng-90:,3:4].values


    y = (a+b)/2
    
    from sklearn.linear_model import LinearRegression
    linear_reg=LinearRegression()
    linear_reg.fit(x,y)


    #fitting polynomial regression
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg=PolynomialFeatures(degree=20) #change only this degree value to change the degree of eqution
    x_poly=poly_reg.fit_transform(x)#also includes 1 in beginning
    linear_reg2=LinearRegression()
    linear_reg2.fit(x_poly,y)


    st.markdown("##")
    #visualizing linear regression result
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;"> Visualizing Linear Regression Results </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    plt.figure(figsize = (10, 4))
    plt.plot(x , y , color='red')
    plt.plot(x, linear_reg.predict(x) , color='blue')
    plt.ylabel('Mid price')
    st.pyplot()

    st.markdown("##")
    #visualizing polynomial regression result
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;"> Visualizing Polynomial Regression Results </h3>'
    st.markdown(graph, unsafe_allow_html=True)
    plt.figure(figsize = (10, 4))
    x_grid= np.arange(min(x), max(x), 0.1)
    x_grid=np.reshape(x_grid, (len(x_grid),1))
    # above 2 lines are used to make graph continuous
    plt.plot(x , y , color='red')
    plt.plot(x_grid , linear_reg2.predict(poly_reg.fit_transform(x_grid)) , color='blue')
    plt.ylabel('Mid price')
    st.pyplot()


    # # predicting new results using linear regression
    # y_1 = linear_reg.predict([[6.5]])
    # print(y_1)

    matrix = []

    for r in range(0, 15):
        matrix.append([r+leng for c in range(0, 1)])

    st.markdown("##")
    # predicting new results using polynomial regression
    graph = '<h3 style="font-family:sans serif; color:#9966cc; font-size: 25px; font-style: italic;">Visualizing Future Prediction Using Polynomial Regression</h3>'
    st.markdown(graph, unsafe_allow_html=True)
    plt.figure(figsize = (10, 4))
    y2= linear_reg2.predict(poly_reg.fit_transform(matrix))
    matrix=np.array(matrix)
    #visualizing polynomial regression result
    x_grid= np.arange(min(matrix), max(matrix), 0.1)
    x_grid= np.reshape(x_grid, (len(x_grid),1))
    # above 2 lines are used to make graph continuous
    # plt.scatter(x , y , color='red')
    
    plt.plot(x_grid , linear_reg2.predict(poly_reg.fit_transform(x_grid)) , color='blue')
    plt.ylabel('Mid price')
    st.pyplot()

    
if __name__ == "__main__":
    title = '<h2 style="font-family:sans serif; color:#90EE90; font-size: 50px;">Stock Market Prediction </h2>'
    st.markdown(title, unsafe_allow_html=True)
    st.image('stock.png', width = 520)
    st.markdown("##")
    page = st.sidebar.radio("Navigation Bar", ("Home", "About Us"))
    st.sidebar.markdown("##")
    if page == "Home":
        var = st.sidebar.selectbox("Which Model you want to use?", ('Polynomial Regression', 'Random Forest', 'LSTM'))
        if var == "Random Forest":
            stock = st.selectbox("Which Stock you want to check?", ('NIFTY 50', 'TESLA', 'AMAZON', 'APPLE', 'Madras Rubber Factory (MRF)', 'State Bank of India (SBI)', 'TATA Consulting Services (TCS)'))
            if stock == "APPLE":
                func("AAPL")
            if stock == "TESLA":
                func("TSLA")
            if stock == "AMAZON":
                func("AMZN")
            if stock == "NIFTY 50":
                func("^NSEI")
            if stock == 'Madras Rubber Factory (MRF)':
                func('MRF.NS')
            if stock == 'State Bank of India (SBI)':
                func('SBIN.NS')
            if stock == 'TATA Consulting Services (TCS)':
                func('TCS.NS')
                
        if var == "LSTM":
            stock = st.selectbox("Which Stock you want to check?", ('NIFTY 50', 'TESLA', 'AMAZON', 'APPLE', 'Madras Rubber Factory (MRF)', 'State Bank of India (SBI)', 'TATA Consulting Services (TCS)'))
            if stock == "APPLE":
                funcLSTM("AAPL")
            if stock == "TESLA":
                funcLSTM("TSLA")
            if stock == "AMAZON":
                funcLSTM("AMZN")
            if stock == 'NIFTY 50':
                funcLSTM('^NSEI')
            if stock == 'Madras Rubber Factory (MRF)':
                funcLSTM('MRF.NS')
            if stock == 'State Bank of India (SBI)':
                funcLSTM('SBIN.NS')
            if stock == 'TATA Consulting Services (TCS)':
                funcLSTM('TCS.NS')
                
        if var == "Polynomial Regression":
            stock = st.selectbox("Which Stock you want to check?", ('NIFTY 50', 'TESLA', 'AMAZON', 'APPLE', 'Madras Rubber Factory (MRF)', 'State Bank of India (SBI)', 'TATA Consulting Services (TCS)'))
            if stock == "APPLE":
                funcPoly("AAPL")
            if stock == "TESLA":
                funcPoly("TSLA")
            if stock == "AMAZON":
                funcPoly("AMZN")
            if stock == 'NIFTY 50':
                funcPoly('^NSEI')
            if stock == 'Madras Rubber Factory (MRF)':
                funcPoly('MRF.NS')
            if stock == 'State Bank of India (SBI)':
                funcPoly('SBIN.NS')
            if stock == 'TATA Consulting Services (TCS)':
                funcPoly('TCS.NS')
                
                
    if page == "About Us":
        st.header("Designed by :")
        st.markdown("##")
        st.markdown("##")
        col1, col2, col3 = st.columns(3)
        col1.subheader("Ayush Gupta")
        col1.markdown("<h5> NIT Jalandhar </h5>", unsafe_allow_html=True)
        
        col2.subheader("Amandeep")
        col2.markdown("<h5> NIT Jalandhar </h5>", unsafe_allow_html=True)
        
        col3.subheader("Rajendra Prasad")
        col3.markdown("<h5> NIT Jalandhar </h5>", unsafe_allow_html=True)