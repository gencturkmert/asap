from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer, RobustScaler
import numpy as np

def z_score(train,val,test):
    if len(train) > 0 and len(val) > 0 and len(test) > 0:
        scaler = StandardScaler()
        scaler.fit(train,None)
        return (scaler.transform(train), scaler.transform(val), scaler.transform(test))
    else:
        return train, val, test

def min_max(train,val,test):
    if len(train) > 0 and len(val) > 0 and len(test) > 0:
        scaler = MinMaxScaler()
        scaler.fit(train,None)
        return (scaler.transform(train), scaler.transform(val), scaler.transform(test))
    else:
        return train, val, test

def log_scaling(train, val, test):
    return (np.log10(train + 1), np.log10(val + 1), np.log10(test + 1))

def batch_norm():
    global NORMALIZATION_LAYER
    NORMALIZATION_LAYER = 'batch_norm'
    return

def layer_norm():
    global NORMALIZATION_LAYER
    NORMALIZATION_LAYER = 'layer_norm'
    return

def instance_norm():
    global NORMALIZATION_LAYER
    NORMALIZATION_LAYER = 'instance_norm'
    return

def group_norm():
    global NORMALIZATION_LAYER
    NORMALIZATION_LAYER = 'group_norm'
    return

def box_cox(train,val,test):
    if len(train) > 0 and len(val) > 0 and len(test) > 0:
        transformer = PowerTransformer(method = 'box-cox')
        transformer.fit(train,None)
        #print("fit passed")
        return (transformer.transform(train+1), transformer.transform(val+1),transformer.transform(test+1))
    else:
        return train, val, test
    
def yeo_johnson(train,val,test):
    if len(train) > 0 and len(val) > 0 and len(test) > 0:
        transformer = PowerTransformer(method = 'yeo-johnson')
        transformer.fit(train,None)
        return (transformer.transform(train), transformer.transform(val),transformer.transform(test))
    else:
        return train, val, test


def robust_scaling(train,val,test, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0)):
    if len(train) > 0 and len(val) > 0 and len(test) > 0:
        scaler = RobustScaler(with_centering=with_centering, with_scaling=with_scaling, quantile_range=quantile_range)
        scaler.fit(train,None)
        return (scaler.transform(train), scaler.transform(val),scaler.transform(test))
    else:
        return train, val, test