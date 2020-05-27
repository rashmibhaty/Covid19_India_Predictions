# PCA

# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

#from keras.layers.advanced_activations import LeakyReLU
#from keras.wrappers.scikit_learn import KerasRegressor
from datetime import datetime
from datetime import timedelta
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.dates as mdates 

# Importing the INDIA dataset
dataset_india = pd.read_csv('covid_19_india.csv', parse_dates=[0],date_parser=lambda x: datetime.strptime(x, '%d/%m/%y'),usecols=[1,8])
gk_india = dataset_india.groupby("Date", as_index=False)["Confirmed"].sum() 

country="India"
dataset=dataset_india
gk=gk_india
case_row=2 #one for country
Pol_Degree=9

gk.insert(0, 'New_ID', range(1, 1 + len(gk)))

X = gk.iloc[:,0].values.reshape(-1,1)
y = gk.iloc[:,case_row].values.reshape(-1,1)

#Create polynomial features
poly_reg = PolynomialFeatures(degree = Pol_Degree)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)

# Now let's make the ANN!
from sklearn.preprocessing import StandardScaler
scalerX = StandardScaler().fit(X_poly)
scalery = StandardScaler().fit(y)

X_scaled = scalerX.transform(X_poly)
y_scaled = scalery.transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size = 0.2, random_state = 0)

def baseline_model():
    # Initialising the ANN
    model = Sequential()
    
    # Adding the input layer and the first hidden layer
    model.add(Dense(output_dim = Pol_Degree+1, kernel_initializer='normal',activation = 'relu', input_dim = Pol_Degree+1))
       
    # Adding the second hidden layer
    model.add(Dense(output_dim = Pol_Degree+1, kernel_initializer='normal',activation = 'relu'))
    #model.add(LeakyReLU(alpha=0.1))
    
    # Adding the output layer
    model.add(Dense(output_dim = 1,kernel_initializer='normal',activation = 'linear'))
    
    # Compiling the ANN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    return model

# Fitting the ANN to the Training set
model=baseline_model()
model.fit(X_train, y_train, batch_size = 8, nb_epoch = 200)
# Predicting the Test set results
y_pred = model.predict(X_test)
y_test_inverse = scalery.inverse_transform(y_test)
y_pred_inverse = scalery.inverse_transform(y_pred)

#Save the prediction for next 15 days in excel
original_len = gk.shape[0]
str_last=str(gk['Date'][original_len-1].date())
i=0
for i in range(int(X[0])-1,original_len+15):
    predicted_val=model.predict(scalerX.transform(poly_reg.fit_transform([[i+1]])))
    predicted_val_inverse = scalery.inverse_transform(predicted_val)
    print(int(predicted_val_inverse))
    if i >= original_len :
        gk.loc[i]=[i+1, datetime.strptime(str_last, '%Y-%m-%d')+timedelta(days=i-original_len+1),-1,-1]
    gk.loc[i,"Prediction_ANN"]=int(predicted_val_inverse)
    
gk.to_csv('India_ANN_Prediction.csv') 
   

#Save the prediction graph
X_date = gk.iloc[:,1].values.reshape(-1,1)
y = gk.iloc[:,2].values.reshape(-1,1)
y_pred = gk.iloc[:,3].values.reshape(-1,1)


axes = plt.gca()

date_form = mdates.DateFormatter("%d-%m-%Y")
axes.xaxis.set_major_formatter(date_form)

locator = mdates.DayLocator(interval=3)
axes.xaxis.set_major_locator(locator)

X_date_h=[]
X_date_p=[]
y_h=[]
y_p=[]

for i in range (0,len(y)):
    if( y[i]!=-1):
        X_date_h.append(X_date[i])
        X_date_p.append(X_date[i])
        y_h.append(y[i])
        y_p.append(y_pred[i])
    else:
        X_date_p.append(X_date[i])
        y_p.append(y_pred[i])
        
      
plt.plot(X_date_p, y_p, color = 'red', label='Predicted Values')
plt.scatter(X_date_h, y_h, color = 'purple', label='Data Points')

plt.gcf().autofmt_xdate()

plt.title('Total Covid-19 Confirmed Cases vs Date in '+str(country)+' --- ANN',fontsize=20, fontweight='bold')
plt.xlabel('Date',fontsize=16, fontweight='bold')
plt.ylabel('Total Covid-19 Confirmed Cases in '+str(country),fontsize=16, fontweight='bold')
plt.legend()

figure = plt.gcf()  # get current figure
figure.set_size_inches(32, 18) # set figure's size manually to your full screen (32x18)
plt.savefig(country+"ANN_"+str(Pol_Degree)+'_Predict.png', bbox_inches='tight') # bbox_inches removes extra white spaces
plt.clf()

