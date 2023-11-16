# chao
# 时间：2023/11/14 15:24

import torch
from torch.distributions import multinomial #概率分布包中的多项分布
from d2l import torch as d2l


fair_probs = torch.ones([6]) / 6
counts = multinomial.Multinomial(1000, fair_probs).sample() #抽1000个样本，查看概率分布
print('掷骰子1000次，查看每个面的相对概率', counts / 1000)

#设置500组抽样，每组抽样100次
counts_500 = multinomial.Multinomial(10, fair_probs).sample((500,))
print(counts_500.size()) #得到一个(500， 6)的概率分布
cum_counts_500 = counts_500.cumsum(axis=0)
print(cum_counts_500.size())
estimates = cum_counts_500 / cum_counts_500.sum(dim=1, keepdims=True)
print(estimates)

# 绘制概率曲线
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
d2l.plt.show()