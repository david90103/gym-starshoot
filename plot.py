import matplotlib.pyplot as plt

with open('reward.txt', 'r') as f:
    res = f.readlines()

res = list(map(lambda x: float(x), res))

plt.title('DQN2')
plt.grid(alpha=0.3)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.plot(res, 'c')
plt.savefig('reward.png')
