from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import math
import os

app = Flask(__name__)

def convolution(signal1, signal2, dt):
    convolved_signal = np.zeros(len(signal1) + len(signal2) - 1)
    t_conv = np.arange(0, len(convolved_signal) * dt, dt)
    for i in range(len(signal1)):
        for j in range(len(signal2)):
            convolved_signal[i + j] += signal1[i] * signal2[j] * dt
    return convolved_signal, t_conv

def generate_signal(signal_type, t, label, user_start=None, user_end=None):
    if signal_type == 'exp':
        return [math.exp(-val) for val in t]
    elif signal_type == 'sin':
        return [math.sin(2 * math.pi * val) for val in t]
    elif signal_type == 'cos':
        return [math.cos(2 * math.pi * val) for val in t]
    elif signal_type == 'user':
        return [1 if user_start <= val <= user_end else 0 for val in t]
    else:
        raise ValueError("Invalid signal type selected.")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        x_type = request.form.get("x_type")
        y_type = request.form.get("y_type")
        TS = float(request.form.get("t_start"))
        TE = float(request.form.get("t_end"))
        t = np.linspace(TS, TE, 2002)
        dt = t[1] - t[0]

        x_start = float(request.form.get("x_start") or TS)
        x_end = float(request.form.get("x_end") or TE)
        y_start = float(request.form.get("y_start") or TS)
        y_end = float(request.form.get("y_end") or TE)

        x = generate_signal(x_type, t, 'x', x_start, x_end)
        y = generate_signal(y_type, t[:-1], 'y', y_start, y_end)
        conv, t_conv = convolution(x, y, dt)

        # Plot
        plt.figure(figsize=(10, 8))
        plt.subplot(3, 1, 1)
        plt.plot(t, x)
        plt.title(f'x(t) - {x_type}')
        plt.subplot(3, 1, 2)
        plt.plot(t[:-1], y)
        plt.title(f'y(t) - {y_type}')
        plt.subplot(3, 1, 3)
        plt.plot(t_conv, conv)
        plt.title('Convolution Result')
        plt.tight_layout()
        plot_path = os.path.join("static", "plot.png")
        plt.savefig(plot_path)
        plt.close()
        return render_template("index.html", plot=True)

    return render_template("index.html", plot=False)

# ✅ Add this to run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
