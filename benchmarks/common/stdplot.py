import torch
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

params = {
    "axes.titlesize": 22,
    "legend.fontsize": 16,
    "figure.figsize": (16, 10),
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 22
}

plt.rcParams.update(params)
plt.style.use("seaborn-whitegrid")
sns.set_style("white")

variable = "tmax"

files = [
    (f"results/convcnp/{variable}/std.{variable}.pkl", "ConvCNP"),
    (f"results/convlnp/{variable}/std.{variable}.pkl", "ConvLNP-UNI"),
    (f"results/convlnp/multivariate/std.{variable}.pkl", "ConvLNP-MUL"),
    (f"results/tnpa/{variable}/std.{variable}.pkl", "TNPA-UNI"),
    (f"results/tnpa/multivariate/std.{variable}.pkl", "TNPA-MUL"),
    (f"results/tnpd/{variable}/std.{variable}.pkl", "TNPD-UNI"),
    (f"results/tnpd/multivariate/std.{variable}.pkl", "TNPD-MUL"),
]

df = pd.DataFrame(columns = ["Experiment", "Uncertainty"])
for index, (file, experiment) in enumerate(files):
    cdf = pd.DataFrame(columns = ["Experiment", "Uncertainty"])
    cdf["Uncertainty"] = pickle.load(open(file, "rb")).flatten()
    cdf["Experiment"] = experiment
    cdf = cdf.loc[(cdf["Uncertainty"] > cdf["Uncertainty"].quantile(0.02)) & (cdf["Uncertainty"] < cdf["Uncertainty"].quantile(0.98))]
    df = pd.concat([df, cdf])
df["Uncertainty"] = pd.to_numeric(df["Uncertainty"])

ax = plt.axes()
plt.figure(figsize = (13, 10), dpi = 80)
plot = sns.violinplot(x = "Experiment", y = "Uncertainty", data = df, scale = "width", inner = "quartile", palette = "Pastel1")
plot.set_xticklabels(plot.get_xticklabels(), rotation = 30)
plot.set_ylabel("Standard deviation", fontsize = 18, labelpad = 12)
d = {"tmax": "Max Temperature", "prep": "Precipitation"}
plt.title(f"Uncertainty in {d[variable]}", fontsize = 22)
plt.savefig(f"plots/{variable}.png")