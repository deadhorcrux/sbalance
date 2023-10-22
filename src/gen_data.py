

import pandas as pd

if __name__ == '__main__':
    
    syed_f = pd.read_csv('../data/syed_factors.csv')
    syed_p = pd.read_csv('../data/syed_pre_post.csv')
    mi  = pd.read_csv('../data/mi.csv')
    new = pd.read_excel('../data/new.xlsx')
    
    feat = ['ODI-preop', 'PT-preop', 'SS-preop', 'PI-preop', 'LL-preop', 'LL-postop']
    df_new = new[feat]

    hasan = pd.read_excel('../data/hasan.xlsx')
    
    rename_dict_n = dict(zip(df_new.columns, [
        'ODI',
        'PT',
        'SS',
        'PI',
        'LL',
        'target',
    ]))

    df_new = df_new.rename(columns=rename_dict_n)

    df_new = df_new.dropna()

    feat_p = ['ll', 'll-p', 'PT', 'PI', 'SS']
    feat_f = ['Pre op ODI']

    df_s = pd.concat([syed_p[feat_p], syed_f[feat_f]], axis=1)
    

    rename_dict_s = dict(
        zip(df_s.columns,
            [
                'LL',
                'target',
                'PT',
                'PI',
                'SS',
                'ODI',
            ]
       )
)
    df_s.rename(columns=rename_dict_s, inplace=True)

    feat = ['preop.1', 'preop.2', 'preop.3', 'preop.4', 'preop.5', 'postop.3']
    df_m = mi[feat].copy()
    rename_dict_m = dict(
        zip(df_m.columns,
            [
                'ODI',
                'PT',
                'SS',
                'PI',
                'LL',
                'target',
            ]))
    df_m.rename(columns=rename_dict_m, inplace=True)
    
    df = pd.concat([df_m, df_s, df_new], axis=0)
    df.reset_index(drop=True, inplace=True)

    df.to_pickle('../data/trainV1.pickle')

    df['PI-LL'] = df['PI'] - df['LL']
    df['target'] = df['PI'] - df['target']

    df = df.drop(['PI', 'LL', 'SS'], axis=1)

    df = pd.concat([df, hasan[['ODI', 'PT', 'PI-LL', 'target']]])
    df.reset_index(drop=True, inplace=True)
    
    df.to_pickle('../data/trainV2.pickle')
    
