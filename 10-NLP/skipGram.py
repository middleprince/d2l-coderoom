import collections
import math
import random
import sys
import time
import zipfile
import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn


# @fucntion: get_centers_and_contexts. 从数据集中建立中心词与背景词
# @params: dataset(整个词典）, max_window_size（每个中心词的window最大尺寸）
# @return: centers, contexts 中心词list及其对以的背景词contexts list
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, center_i - window_size), 
                                 min(len(st), center_i + 1 + window_size)))
            indices.remove(center_i)
            contexts.append([st[idx] for idx in indices])
    return centers, contexts
                           

# @function: get_negatives. 实现负采样，重点获取一口contex window中K 个噪声词
# @params: all_contexts（所有的contexts词语为idx二维list）, 
#          sampling_weights（噪声词采样词典中各个词的权重）, 
#          K（一个window窗口中噪声词的数量）
# @return: all_negatives. 返回负采样后的数据，整个是二维的的数据
def get_negatives(all_contexts, sampling_weights, K):
    all_negtives, neg_condidates, i= [], [], 0
    polulation = list(range(len(sampling_weights)))
    for contexts in all_contexts:
        negtives = []
        while len(negtives) < len(contexts) * K:
            if i == len(neg_condidates):
                i, neg_condidates = 0, random.choices(
                    polulation, sampling_weights, k=int(1e5))
            neg, i = neg_condidates[i], i+1
            if neg not in set(contexts):
                negtives.append(neg)
        all_negtives.append(negtives)
    return all_negtives



# subsampling to remove some redundent words
def discard(idx):
    return (random.uniform(0, 1) < 
            1- math.sqrt(1e-4 / counter[idx_to_token[idx]] * num_tokens))

subsampled_dataset = [[tk for tk in st if not discard(tk)] 
                        for st in dataset]

# @function: batchfy. 对数据进行batch化处理,其中所有的context加negative
# 会做定长度的处理，最大长度下做mask的处理来标示。
# @para: data, 整个高训练的数据集包含：centers、contexts、negative
# @return: centers（中心词的向量，二维）, 
#          context_negatives（补全到最大长度的context与negatives，二维,
#          mask（有效词掩码，二维）, labels（context词语掩码，二维—）
def batchify(data):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, context_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        context_negatives += [context + negative + [0]*(max_len -cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (nd.array(centers).reshape((-1, 1)), nd.array(context_negatives),
            nd.array(masks), nd.array(labels))

# SKIP Gram
# @function: skip_gram. skipGram模型
# @param: center, contexts_and_negativees, ebed_v, embed_u
# @return: pred
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = nd.batch_dot(v, u.swapaxes(1,2))
    return pred

# @function: train. 模型训练模块
# @param: net(预先定义好的网络), lr(学习率), num_epochs(训练的epochs)
# @return: void.
def train(net, lr, num_epochs):
    ctx = d2l.try_gpu()
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate' : lr})
    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in data_iter:
            center, context_negative, mask, label = [data.as_in_context(ctx) 
                    for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                  mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            l_sum += l.sum().asscalar()
            n += l.size
        print('epoch %d, loss %.2f, %.2fs' % (epoch + 1, l_sum / n, time.time() - start))
        
# @function:get_similar_tokens. 从所有的词典中找出最相似的k个近义词
# @param: query_token(待查询的词汇), k(top k 个词), embed(学习完成的embedding）
# @return: void
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W(token_to_idx[query_token])
    cos = nd.dot(W, x) / (nd.sum(W*W, axis=1) * nd.sum(x * x) + 1e-9).sqrt()
    topk = nd.topk(cos, k=k+1, ret_type='indices').asnumpy().astype('int32')
    for i in topk[1:]:
        print('cosine sim=%.3f: %s ' % (cos[i].asscalar(), (idx_to_token[i])))

if __name__ == 'main':
    with zipfile.ZipFile('../../d2l-zh/data/ptb.zip', 'r') as zin:
        zin.extractall('../../d2l-zh/data')
    
    with open('../../d2l-zh/data/ptb/ptb.train.txt', 'r') as f:
        lines = f.readlines()
        raw_dataset = [st.split() for st in lines]
    
    # build trainning dictionary
    counter = collections.Counter([tk for st in raw_dataset for tk in st])
    counter = dict(filter((lambda x: x[1]>5), counter.items()))
    
    idx_to_token = [tk for tk, _ in counter.items()]
    token_to_idx = {tk: idx for idx, tk in enumerate(idx_to_token)}
    dataset = [[token_to_idx[tk] for tk in st if tk in token_to_idx] 
               for st in raw_dataset]
    num_tokens = sum(len(st) for st in dataset)

    all_centers, all_contexts = get_centers_and_contexts(subsampled_dataset, 5)
    #论文中建议的噪声词的采样频率与总词频的0.75次方
    sampling_weights = [counter[w]**0.75 for w in idx_to_token] 
    all_negatives = get_negatives(all_contexts, sampling_weights, 5)
    
    #get data iterator
    batch_size = 512
    num_workers = 0 if sys.platform.startswith('win32') else 4
    dataset = gdata.ArrayDataset(all_centers, all_contexts, all_negatives)
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True,
            batchify_fn=batchify, num_workers=num_workers)
    # constructing the model
    embed_size = 100
    net = nn.Sequential()
    net.add(nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size),
            nn.Embedding(input_dim=len(idx_to_token), output_dim=embed_size))
    loss = gloss.SigmoidBinaryCrossEntropyLoss() 
    
    train(net, 0.005, 5) 
    
    get_similar_tokens('justice', 3, net[0])
