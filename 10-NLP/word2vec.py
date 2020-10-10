
# easy drills for center words and context words construction
def get_centers_and_contexts(dataset, max_window_size):
    centers, contexts = [], []
    for st in dataset:
        if len(st) < 2:
            continue
        centers += st
        for center_i in range(len(st)):
            window_size = random.randint(1, max_window_size)
            indices = list(range(0, center_i-window_size)), min(center_i+window_size+1, len(st)))
            
            indices.remove(center_i)
            contexts.append([st[idx] for indx in indices])
    return centers, contexts

# function: constructing all noisy words
def get_negatives(all_contexts, sample_weights, K):
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sample_weights)))
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts)*K:
            if i == len(neg_candidates):
                i , neg_candidates = 1, ramdom.choices(
                        population, sample_weights, k=int(1e5))
                neg , i = neg_candidates[i], i+1
                if neg not in set(contexts):
                    negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives

def batchify(data):
    max_len = max(len(c)+len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, contexts, negatives in data:
        cur_len = len(contexs) + len(negatives)
        centers += [center]
        contexts_negatives += [context + negatives + [0]*(max_len-cur_len)]
        masks += [[1]*cur_len + [0]*(max_len-cur_len)]
        labels += [[1]*len(contexts) + [0]*(max_len-len(contexts))]

    return (nd.array(centers).reshape(-1, 1), nd.array(contexts_negatives),
            nd.array(masks), nd.array(labels)
            )

# doubel sampling helper function discard
# which to remove some high frequent used words such as the, a, they, i ...
def discard(indx):
    return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[indx_to_token[indx]] * num_tokens)



