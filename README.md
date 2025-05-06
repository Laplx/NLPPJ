# NLPPJ

| Model | Sen-Making | Explanation |
| :---: | :--------: | :---------: |
| BERT  |   69.57%   |   37.41%    |
| ELMo  |    62%     |     40%     |
| GPT-2 |   69.97%   |   36.07%    |

sk-e4907720e8e14167ae9a63b8397a95df

$\begin{aligned} & \text { score }_\text{BERT}=\left(p_{w_1} * p_{w_2} * \ldots * p_{w_n}\right)^{(-1 / n)}= \\ & \quad\left(\prod_{i=1}^n P\left(w_i \mid w_1 \ldots w_{i-1} w_{i+1} \ldots w_n\right)\right)^{-1 / n}\end{aligned}$

$\begin{aligned} & \text { score }_\text{GPT}=\left(p_{w_1} * p_{w_2} * \ldots * p_{w_n}\right)^{(-1 / n)}= \\ & \quad\left(\prod_{i=1}^n P\left(w_i \mid w_1 \ldots w_{i-1} \right)\right)^{-1 / n} = P(w_1 \ldots w_{n})^{-1 / n}  \end{aligned}$



### ELMO 比较方式

| 标准名称           | 核心判断依据                           | 计算公式                                                     |
| :----------------- | :------------------------------------- | :----------------------------------------------------------- |
| L2范数比较         | ELMo向量的欧几里得长度                 | np．linalg．norm（emb）                                      |
| 与零向量余弦相似度 | 句向量与零向量的方向一致性             | $\cos (\theta)=(\mathrm{emb} \cdot \theta) /(\|\mathrm{emb}\|\|\theta\|)$ |
| 双向LSTM余弦相似度 | Forward／Backward LSTM向量的方向一致性 | $\cos (\mathrm{fw}, \mathrm{bw})$                            |
| 双向LSTM欧氏距离   | Forward／Backward LSTM向量的几何距离   | $\|f w-b w\| 2$                                              |

### KG 微调

- NER + Concat + Score

- 
