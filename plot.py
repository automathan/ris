
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('results/progress.csv')
plt.plot(data['rollout/ep_rew_mean'])
plt.show()