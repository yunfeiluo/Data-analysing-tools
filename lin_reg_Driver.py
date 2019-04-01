import lin_reg as lg
import numpy as np
import pandas as pd

def main():
    #np.random.seed(0)

    x_data = np.linspace(-5, 5, num=50)
    y_data = np.random.random()*x_data + np.random.normal(size=50)
    #y_data = np.random.normal(size=50)
    #z_data = np.random.normal(size=50)
    z_data = np.random.random()*x_data + np.random.random()*y_data + np.random.normal(size=50)
    output = np.random.random()*z_data + np.random.random()*x_data + np.random.random()*y_data + np.random.normal(size=50)

    df = {}
    df["x1"] = x_data
    df["x2"] = y_data
    df["x3"] = z_data
    df["x4"] = output
    df = pd.DataFrame(data=df)
    print(df.corr())    

    
    #lg.plot3var(df)
    #lg.multiVarLinReg(lg.lin_reg2d, [df.iloc[:, 0], df.iloc[:, 1]], "2d")
    #lg.multiVarLinReg(lg.lin_reg3d, [df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]], "3d")
    #lg.multiVarLinReg(lg.lin_reg4d, [df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3]], "4d")

if __name__ == '__main__':
    main()