import numpy as np
import matplotlib.pyplot as plt


x = np.array([1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7500, 9000])
y = np.array([200, 300, 400, 600, 800, 1000, 1200, 1500, 1800, 2200, 2700])

x_normalized = (x - np.mean(x)) / np.std(x)
y_normalized = (y - np.mean(y)) / np.std(y)

w = 0  #weight... slope of the line (parameters of equations)
b = 0  #bias....Y inttercept of line (paramenters of equations(2))
learning_rate = 0.01
iterations = 2000

def compute_cost(x, y, w, b):
    m = len(x)
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b  #y= mx + b
        cost += (f_wb - y[i]) ** 2  #(Y - Y')^2 / 2m
    return cost / (2 * m)

def gradient_descent(x, y, w, b, learning_rate, iterations):    #differential part of the equation to optimize the output....
    m = len(x)
    for it in range(iterations):
        dw = 0 
        db = 0
        for i in range(m):
            f_wb = w * x[i] + b
            dw += (f_wb - y[i]) * x[i]   #for predicting or optimizing the w...> (Y - Y')^2 * x (iterating input)
            db += (f_wb - y[i])          #same formula but no x
        dw /= m                          # dw / m.... where m is total number of input(x)
        db /= m
        w -= learning_rate * dw
        b -= learning_rate * db
        if it % 100 == 0:                #refresh the results after every 100 iteration
            cost = compute_cost(x, y, w, b)
            print(f"Iteration {it}: Cost {cost:.2f}, w {w:.2f}, b {b:.2f}")
    return w, b

def model(x, w, b):
    return w * x + b                   #main y= mx+b equation

def linear_regression(x, y, learning_rate=0.01, iterations=2000):
    w, b = 0, 0
    w, b = gradient_descent(x_normalized, y_normalized, w, b, learning_rate, iterations)
    print(f"Optimized w: {w:.2f}, Optimized b: {b:.2f}")

    predicted_y = model(x_normalized, w, b)  #recalling functions

    predicted_y_rescaled = predicted_y * np.std(y) + np.mean(y)  #rescaling the inputs for bigger values

    plt.scatter(x, y, marker='x', c='r', label='Actual Values')
    plt.plot(x, predicted_y_rescaled, c='b', label='Model Prediction')
    plt.title("Linear Regression Model")
    plt.xlabel("Input Feature (sq ft)")
    plt.ylabel("Target Value (in thousands)")
    plt.legend()
    plt.show()
    return w, b

linear_regression(x, y)
