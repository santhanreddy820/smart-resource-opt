import pandas as pd

def create_features(df):
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])
    df = df.sort_values(['device_id','ts'])

    df['hour'] = df['ts'].dt.hour
    df['dow'] = df['ts'].dt.dayofweek
    df['is_weekend'] = (df['dow'] >= 5).astype(int)

    df['power_roll_1h'] = df.groupby('device_id')['power_kW'] \
        .transform(lambda x: x.rolling(4, min_periods=1).mean())

    df['power_roll_6h'] = df.groupby('device_id')['power_kW'] \
        .transform(lambda x: x.rolling(24, min_periods=1).mean())

    df['power_lag_1'] = df.groupby('device_id')['power_kW'].shift(1)
    df['power_lag_4'] = df.groupby('device_id')['power_kW'].shift(4)

    df['power_delta_24h'] = df['power_kW'] - df.groupby('device_id')['power_kW'] \
        .transform(lambda x: x.shift(96).rolling(96, min_periods=1).mean())

    df.fillna(0, inplace=True)
    return df
