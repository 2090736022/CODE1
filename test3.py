import tushare as ts
pro = ts.pro_api('a14f37a303d0e280c839224d331315323a8cde36ffbe5c41757fef31')
df = pro.moneyflow(ts_code='002421.SZ', start_date='20260605', end_date='20260605')
print(df)