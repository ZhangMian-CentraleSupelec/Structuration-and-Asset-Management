# Structuration-and-Asset-Management
- This project is based on the course in 3A CentraleSupélec, Structuration and Asset Management. 
- The data of this project is based on the historic data of the top 10 biggest companies in NYSE.

### What we have reached

The historical return based on the different trading strategies. 

![image](https://user-images.githubusercontent.com/110284601/185870692-67b98c7c-6a4b-4cc5-b60c-b40f12d2eb75.png)



## 1. Import data and plot the historical performance

  - The historical return for the stocks.
  
  ![image](https://user-images.githubusercontent.com/110284601/185870972-49c7a0c6-1a2a-4235-9bf9-1adabe060a5a.png)

  - Our strategy
  
    We split the dataset in four periods:

    -a one year selection period, used to select the 5 assets

    -the rest of the dataset, used to backtest the dynamic strategies

    -a training period, used to chose the weights of the static portfolio

    -a testing period, used to assess the performances of the static portfolio


## 2. Data selection
  
  - We select the combination of 5 assets that has the lowest average covariance among the 100 combinations that have the highest return on average over the selection period. 

```
import itertools

indexes = list(itertools.combinations(range(0,return_selection.shape[1]),5))
rets_to_beat = []
inds = []
cov_to_beat = 1
i = 0
n_max = 100
for index in indexes:
    returns = return_selection.iloc[:,list(index)]
    ret = np.mean(returns.to_numpy())
    co = np.cov(returns.to_numpy(),rowvar = False,bias = True)
    co = np.sum(co) - np.sum(np.diag(co))
    if len(rets_to_beat) < n_max:
        rets_to_beat += [ret]
        if co<cov_to_beat:
            cov_to_beat = co
            i=len(index)
        inds += [index]
    else:
        n = np.argmin(rets_to_beat)
        if rets_to_beat[n] < ret:
            rets_to_beat[n] = ret
            inds[n] = index
            if co<cov_to_beat:
                cov_to_beat = co
                i=n

kept_stocks = [equity_list[inds[i][j]] for j in range(5)]
kept_stocks
['AAPL US Equity',
 'MC FP Equity',
 'BNP FP Equity',
 'XAUEUR Curncy',
 'INDMLTO FP Equity']
```



## 3. Metrics and strategies definition

  - We define 2 main strategies: max sharpe ratio and min variance. We use them with two estimators of the covariance matrix: the historic estimator and the capped estimator, obtained by setting all the eigenvalues of the historic covariance matrix to 0 except for the biggest one.
```
def port_ret(weights, rets_means):
    return np.sum(rets_means * weights) * 126

def port_vol(weights, rets_cov):
    return np.sqrt(126* weights.T @ rets_cov @ weights)

def neg_sharpe_ratio(weights, rets_means, rets_cov):  
    return - port_ret(weights, rets_means) / port_vol(weights, rets_cov)  

```


```
def Max_SR_portfolio(rets_means, rets_cov, range_bnds = (0, 1)):
    nb_equity = len(rets_cov)    
    eweights = np.array(nb_equity * [1. / nb_equity,])  
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})  
    bnds = tuple(range_bnds for x in range(nb_equity))
    objective_SR = lambda x: neg_sharpe_ratio(x, rets_means, rets_cov)
    
    MSR_pf = sco.minimize(objective_SR, eweights, method='SLSQP', bounds=bnds, constraints=cons) 
    
    return MSR_pf, port_vol(MSR_pf['x'], rets_cov), port_ret(MSR_pf['x'], rets_means)

```

```
def MV_portfolio(rets_means, rets_cov, range_bnds = (0, 1)):
    nb_equity = len(rets_cov)    
    eweights = np.array(nb_equity * [1. / nb_equity,])  
    cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})  
    bnds = tuple(range_bnds for x in range(nb_equity))
    
    MV_pf = sco.minimize(port_vol, eweights,rets_cov,
                    method='SLSQP', bounds=bnds, constraints=cons) 
    
    return MV_pf, port_vol(MV_pf['x'], rets_cov), port_ret(MV_pf['x'], rets_means)

```
```
def sigma_c(Sigma):
    # Step 1 : determining Lambda and V
    Lamb = np.diag(np.linalg.eigh(Sigma)[0])
    V = np.linalg.eigh(Sigma)[1]
    # Step 2 : Computing Lambda clipped
    c = np.sum(Lamb.diagonal() * (Lamb.diagonal() < np.max(Lamb))) / (np.sum(Lamb.diagonal() < np.max(Lamb))) * np.ones(len(Lamb.diagonal()))
    Lamb_c = np.diag(c * (Lamb.diagonal() < np.max(Lamb)) + Lamb.diagonal() * (Lamb.diagonal() == np.max(Lamb)))
    # Step 3 : Computing C_t
    C_t = V @ Lamb_c @ V.transpose()
    # Step 4 : Computing C_c
    C_c = np.zeros(shape = (len(C_t), len(C_t)))
    for i in range(len(C_t)):
        for j in range(len(C_t)):
            C_c[i,j] = C_t[i,j]/(C_t[i,i] * C_t[j,j])

    # Step 5 : Computing clipped covariance matrix
    Sigma_in_c = np.zeros(shape = (len(C_t), len(C_t)))
    for i in range(len(C_t)):
        for j in range(len(C_t)):
            Sigma_in_c[i,j] = C_c[i,j] * Sigma[i,i] * Sigma[j,j]
    return Sigma_in_c
```


## 4. Static portfolio

```
weights = np.zeros(5)
gain_MV = []
gain_SR = []
gain_MV_clipped = []
gain_SR_clipped = []
gain_index = []

return_in = return_static_train
return_in_mean = return_in.mean()
return_in_cov = np.cov(return_in, rowvar = False, bias = True)
return_in_cov_clipped = sigma_c(return_in_cov)

return_out = return_static_test
return_out_mean = return_out.mean()
return_out_cov = np.cov(return_out, rowvar = False, bias = True)

nb_equity = len(return_in.T)
weight_SR = Max_SR_portfolio(return_in_mean, return_in_cov, range_bnds = (0, 1))[0]['x']
weight_MV = MV_portfolio(return_in_mean, return_in_cov, range_bnds = (0, 1))[0]['x']
weight_SR_clipped = Max_SR_portfolio(return_in_mean, return_in_cov_clipped, range_bnds = (0, 1))[0]['x']
weight_MV_clipped = MV_portfolio(return_in_mean, return_in_cov_clipped, range_bnds = (0, 1))[0]['x']
eq_weights = np.ones(5)/5

return_port_SR = port_ret(weight_SR, return_out_mean)
return_port_MV = port_ret(weight_MV, return_out_mean)
return_port_SR_clipped = port_ret(weight_SR_clipped, return_out_mean)
return_port_MV_clipped = port_ret(weight_MV_clipped, return_out_mean)
return_index = port_ret(eq_weights, return_out_mean)

vol_port_SR = port_vol(weight_SR, return_out_cov)
vol_port_MV = port_vol(weight_MV, return_out_cov)
vol_port_SR_clipped = port_vol(weight_SR_clipped, return_out_cov)
vol_port_MV_clipped = port_vol(weight_MV_clipped, return_out_cov)
vol_index = port_vol(eq_weights, return_out_cov)

sharpe_SR = return_port_SR/vol_port_SR
sharpe_MV = return_port_MV/vol_port_MV
sharpe_SR_clipped = return_port_SR_clipped/vol_port_SR_clipped
sharpe_MV_clipped = return_port_MV_clipped/vol_port_MV_clipped
sharpe_index = return_index/vol_index
```

## 5. Dynamic portfolio and CPPI Strategy

![image](https://user-images.githubusercontent.com/110284601/185872521-00ee55b7-e131-4035-b871-4f8cf5badb2f.png)

```
print(np.mean(SR_SR),np.mean(SR_MV),np.mean(SR_SR_c),np.mean(SR_MV_c),np.mean(SR_index))
0.6278244513314 0.6809187021467289 0.5193745085032088 0.4608307131658638 0.8113408395999259
```
The equally weighted portfolio and the maximum Sharpe ratio with historic covariance matrix are the best performers in terms of returns. However, the minimum variance portfolio is better than the maximum sharpe ratio in terms of sharpe ratio over the period.

```
def CPPI(equity_price, floor, multiplier):
  equity_value = np.ones(len(equity_price))
  equity_amount = np.ones(len(equity_price))
  total_value = np.ones(len(equity_price)) 
  bond_value = np.ones(len(equity_price))

  equity_amount[0] = (equity_price[0] - floor) * multiplier
  equity_value[0] = equity_price[0] * equity_amount[0]
  bond_value[0] = total_value[0] - equity_amount[0] * equity_value[0]

  for i in range(0, len(equity_price) - 1):
    equity_value[i+1] = multiplier * equity_amount[i] * equity_price[i+1] + bond_value[i] - floor
    equity_amount[i+1] = equity_value[i+1] / equity_price[i+1]
    bond_value[i+1] = equity_amount[i]*equity_price[i+1] + bond_value[i] - equity_value[i+1]
    total_value[i+1] = equity_value[i+1] + bond_value[i+1]
  return total_value
```

![image](https://user-images.githubusercontent.com/110284601/185872751-fd3d344f-9f0e-4674-ad0f-ce80f818b9db.png)



