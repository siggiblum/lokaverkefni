import pandas as pd
listi = [{1:2}, {3:4}]
fm = pd.DataFrame(listi)
print(fm.loc[1])
