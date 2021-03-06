import os
import io
import re
import sys
import pickle
import random
import inspect
import argparse
import subprocess
import numpy as np
import torch
import difflib
from torch import optim
from logging import getLogger

from .logger import create_logger
from .dictionary import Dictionary


MAIN_DUMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'dumped')

logger = getLogger()


# load Faiss if available (dramatically accelerates the nearest neighbor search)
try:
    import faiss
    FAISS_AVAILABLE = True
    if not hasattr(faiss, 'StandardGpuResources'):
        sys.stderr.write("Impossible to import Faiss-GPU. "
                         "Switching to FAISS-CPU, "
                         "this will be slower.\n\n")

except ImportError:
    sys.stderr.write("Impossible to import Faiss library!! "
                     "Switching to standard nearest neighbors search implementation, "
                     "this will be significantly slower.\n\n")
    FAISS_AVAILABLE = False


def initialize_exp(params):
    """
    Initialize experiment.
    """
    # initialization， 用seed来初始化
    if getattr(params, 'seed', -1) >= 0:
        # 保证生成的随机数具有相同的可预测性，即seed相同，每次生成的随机数都是一样的
        np.random.seed(params.seed)
        # 为CPU设置种子用于生成随机数，以使得结果是确定的
        torch.manual_seed(params.seed)
        if params.cuda:
            # 为当前GPU设置随机种子；如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
            torch.cuda.manual_seed(params.seed)

    # dump parameters
    params.exp_path = get_exp_path(params)
    with io.open(os.path.join(params.exp_path, 'params.pkl'), 'wb') as f:
        pickle.dump(params, f)

    # create logger
    logger = create_logger(os.path.join(params.exp_path, 'train.log'), vb=params.verbose)
    logger.info('============ Initialized logger ============')
    logger.info('\n'.join('%s: %s' % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info('The experiment will be stored in %s' % params.exp_path)
    return logger


def load_fasttext_model(path):
    """
    Load a binarized fastText model.
    """
    try:
        import fastText
    except ImportError:
        raise Exception("Unable to import fastText. Please install fastText for Python: "
                        "https://github.com/facebookresearch/fastText")
    return fastText.load_model(path)


def bow(sentences, word_vec, normalize=False):
    """
    Get sentence representations using average bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        sentvec = [word_vec[w] for w in sent if w in word_vec]
        if normalize:
            sentvec = [v / np.linalg.norm(v) for v in sentvec]
        if len(sentvec) == 0:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.mean(sentvec, axis=0))
    return np.vstack(embeddings)


def bow_idf(sentences, word_vec, idf_dict=None):
    """
    Get sentence representations using weigthed IDF bag-of-words.
    """
    embeddings = []
    for sent in sentences:
        sent = set(sent)
        list_words = [w for w in sent if w in word_vec and w in idf_dict]
        if len(list_words) > 0:
            sentvec = [word_vec[w] * idf_dict[w] for w in list_words]
            sentvec = sentvec / np.sum([idf_dict[w] for w in list_words])
        else:
            sentvec = [word_vec[list(word_vec.keys())[0]]]
        embeddings.append(np.sum(sentvec, axis=0))
    return np.vstack(embeddings)


def get_idf(europarl, src_lg, tgt_lg, n_idf):
    """
    Compute IDF values.
    """
    idf = {src_lg: {}, tgt_lg: {}}
    k = 0
    for lg in idf:
        start_idx = 200000 + k * n_idf
        end_idx = 200000 + (k + 1) * n_idf
        for sent in europarl[lg][start_idx:end_idx]:
            for word in set(sent):
                idf[lg][word] = idf[lg].get(word, 0) + 1
        n_doc = len(europarl[lg][start_idx:end_idx])
        for word in idf[lg]:
            idf[lg][word] = max(1, np.log10(n_doc / (idf[lg][word])))
        k += 1
    return idf


def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    if FAISS_AVAILABLE:
        emb = emb.cpu().numpy()
        query = query.cpu().numpy()
        if hasattr(faiss, 'StandardGpuResources'):
            # gpu mode
            res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = 0
            index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
        else:
            # cpu mode
            index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        distances, _ = index.search(query, knn)
        return distances.mean(1)
    else:
        bs = 1024
        all_distances = []
        emb = emb.transpose(0, 1).contiguous()
        for i in range(0, query.shape[0], bs):
            distances = query[i:i + bs].mm(emb)
            best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
            all_distances.append(best_distances.mean(1).cpu())
        all_distances = torch.cat(all_distances)
        return all_distances.numpy()


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in ['off', 'false', '0']:
        return False
    if s.lower() in ['on', 'true', '1']:
        return True
    raise argparse.ArgumentTypeError("invalid value for a boolean flag (0 or 1)")


def get_optimizer(s):
    """
    Parse optimizer parameters.
    Input should be of the form:
        - "sgd,lr=0.01"
        - "adagrad,lr=0.1,lr_decay=0.05"
    """
    if "," in s:
        method = s[:s.find(',')]
        optim_params = {}
        for x in s[s.find(',') + 1:].split(','):
            split = x.split('=')
            assert len(split) == 2
            assert re.match("^[+-]?(\d+(\.\d*)?|\.\d+)$", split[1]) is not None
            optim_params[split[0]] = float(split[1])
    else:
        method = s
        optim_params = {}

    if method == 'adadelta':
        optim_fn = optim.Adadelta
    elif method == 'adagrad':
        optim_fn = optim.Adagrad
    elif method == 'adam':
        optim_fn = optim.Adam
    elif method == 'adamax':
        optim_fn = optim.Adamax
    elif method == 'asgd':
        optim_fn = optim.ASGD
    elif method == 'rmsprop':
        optim_fn = optim.RMSprop
    elif method == 'rprop':
        optim_fn = optim.Rprop
    elif method == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization method: "%s"' % method)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn, optim_params


def get_exp_path(params):
    """
    Create a directory to store the experiment.
    """
    # create the main dump path if it does not exist
    exp_folder = MAIN_DUMP_PATH if params.exp_path == '' else params.exp_path
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    assert params.exp_name != ''
    exp_folder = os.path.join(exp_folder, params.exp_name)
    if not os.path.exists(exp_folder):
        subprocess.Popen("mkdir %s" % exp_folder, shell=True).wait()
    if params.exp_id == '':
        chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
        while True:
            exp_id = ''.join(random.choice(chars) for _ in range(10))
            exp_path = os.path.join(exp_folder, exp_id)
            if not os.path.isdir(exp_path):
                break
    else:
        exp_path = os.path.join(exp_folder, params.exp_id)
        assert not os.path.isdir(exp_path), exp_path
    # create the dump folder
    if not os.path.isdir(exp_path):
        subprocess.Popen("mkdir %s" % exp_path, shell=True).wait()
    return exp_path


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def read_txt_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a text file.
    """
    word2id = {}
    vectors = []

    # load pretrained embeddings
    lang = params.src_lang if source else params.tgt_lang
    emb_path = params.src_emb if source else params.tgt_emb
    #词向量的维度，默认为300
    _emb_dim_file = params.emb_dim
    with io.open(emb_path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        for i, line in enumerate(f):
            if i == 0:
                #词向量的第一行是两个数据，表示的是总共有多少个词向量和词向量的维度
                split = line.split()
                assert len(split) == 2
                assert _emb_dim_file == int(split[1])
            else:
                word, vect = line.rstrip().split(' ', 1)
                if not full_vocab:
                    word = word.lower()
                vect = np.fromstring(vect, sep=' ')
                # 求范数，输入的vect表示举证
                if np.linalg.norm(vect) == 0:  # avoid to have null embeddings
                    vect[0] = 0.01
                if word in word2id:
                    if full_vocab:
                        logger.warning("Word '%s' found twice in %s embedding file"
                                       % (word, 'source' if source else 'target'))
                else:
                    if not vect.shape == (_emb_dim_file,):
                        logger.warning("Invalid dimension (%i) for %s word '%s' in line %i."
                                       % (vect.shape[0], 'source' if source else 'target', word, i))
                        continue
                    assert vect.shape == (_emb_dim_file,), i
                    # 没有出现在word2id中的word会被加入到word2id中。
                    word2id[word] = len(word2id)
                    vectors.append(vect[None])
            # 当不是full_vocab的时候，如果词典的最大数量超过了参数中设置的值，那就不会向词典中添加数据了
            if params.max_vocab > 0 and len(word2id) >= params.max_vocab and not full_vocab:
                break

    assert len(word2id) == len(vectors)
    logger.info("Loaded %i pre-trained word embeddings." % len(vectors))

    # compute new vocabulary / embeddings
    # 将word2id变成id2word
    id2word = {v: k for k, v in word2id.items()}
    #构建字典
    dico = Dictionary(id2word, word2id, lang)
    embeddings = np.concatenate(vectors, 0)
    embeddings = torch.from_numpy(embeddings).float()
    embeddings = embeddings.cuda() if (params.cuda and not full_vocab) else embeddings

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def select_subset(word_list, max_vocab):
    """
    Select a subset of words to consider, to deal with words having embeddings
    available in different casings. In particular, we select the embeddings of
    the most frequent words, that are usually of better quality.
    """
    word2id = {}
    indexes = []
    for i, word in enumerate(word_list):
        word = word.lower()
        if word not in word2id:
            word2id[word] = len(word2id)
            indexes.append(i)
        if max_vocab > 0 and len(word2id) >= max_vocab:
            break
    assert len(word2id) == len(indexes)
    return word2id, torch.LongTensor(indexes)


def load_pth_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a PyTorch binary file.
    """
    # reload PyTorch binary file
    lang = params.src_lang if source else params.tgt_lang
    data = torch.load(params.src_emb if source else params.tgt_emb)
    dico = data['dico']
    embeddings = data['vectors']
    assert dico.lang == lang
    assert embeddings.size() == (len(dico), params.emb_dim)
    logger.info("Loaded %i pre-trained word embeddings." % len(dico))

    # select a subset of word embeddings (to deal with casing)
    if not full_vocab:
        word2id, indexes = select_subset([dico[i] for i in range(len(dico))], params.max_vocab)
        id2word = {v: k for k, v in word2id.items()}
        dico = Dictionary(id2word, word2id, lang)
        embeddings = embeddings[indexes]

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def load_bin_embeddings(params, source, full_vocab):
    """
    Reload pretrained embeddings from a fastText binary file.
    """
    # reload fastText binary file
    lang = params.src_lang if source else params.tgt_lang
    model = load_fasttext_model(params.src_emb if source else params.tgt_emb)
    words = model.get_labels()
    assert model.get_dimension() == params.emb_dim
    logger.info("Loaded binary model. Generating embeddings ...")
    embeddings = torch.from_numpy(np.concatenate([model.get_word_vector(w)[None] for w in words], 0))
    logger.info("Generated embeddings for %i words." % len(words))
    assert embeddings.size() == (len(words), params.emb_dim)

    # select a subset of word embeddings (to deal with casing)
    if not full_vocab:
        word2id, indexes = select_subset(words, params.max_vocab)
        embeddings = embeddings[indexes]
    else:
        word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    dico = Dictionary(id2word, word2id, lang)

    assert embeddings.size() == (len(dico), params.emb_dim)
    return dico, embeddings


def load_embeddings(params, source, full_vocab=False):
    """
    Reload pretrained embeddings.
    - `full_vocab == False` means that we load the `params.max_vocab` most frequent words.
      It is used at the beginning of the experiment.
      In that setting, if two words with a different casing occur, we lowercase both, and
      only consider the most frequent one. For instance, if "London" and "london" are in
      the embeddings file, we only consider the most frequent one, (in that case, probably
      London). This is done to deal with the lowercased dictionaries.
    - `full_vocab == True` means that we load the entire embedding text file,
      before we export the embeddings at the end of the experiment.
    """
    assert type(source) is bool and type(full_vocab) is bool
    emb_path = params.src_emb if source else params.tgt_emb
    if emb_path.endswith('.pth'):
        return load_pth_embeddings(params, source, full_vocab)
    if emb_path.endswith('.bin'):
        return load_bin_embeddings(params, source, full_vocab)
    else:
        return read_txt_embeddings(params, source, full_vocab)


def normalize_embeddings(emb, types, mean=None):
    """
    Normalize embeddings by their norms / recenter them.
    """
    for t in types.split(','):
        if t == '':
            continue
        if t == 'center':
            if mean is None:
                mean = emb.mean(0, keepdim=True)
            emb.sub_(mean.expand_as(emb))
        elif t == 'renorm':
            emb.div_(emb.norm(2, 1, keepdim=True).expand_as(emb))
        else:
            raise Exception('Unknown normalization type: "%s"' % t)
    return mean.cpu() if mean is not None else None


def export_embeddings(src_emb, tgt_emb, params):
    """
    Export embeddings to a text or a PyTorch file.
    """
    assert params.export in ["txt", "pth"]

    # text file
    if params.export == "txt":
        src_path = os.path.join(params.exp_path, 'vectors-Gson.txt')
        tgt_path = os.path.join(params.exp_path, 'vectors-Jackson.txt')
        # source embeddings
        logger.info('Writing source embeddings to %s ...' % src_path)
        with io.open(src_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % src_emb.size())
            for i in range(len(params.src_dico)):
                f.write(u"%s %s\n" % (params.src_dico[i], " ".join('%.5f' % x for x in src_emb[i])))
        # target embeddings
        logger.info('Writing target embeddings to %s ...' % tgt_path)
        with io.open(tgt_path, 'w', encoding='utf-8') as f:
            f.write(u"%i %i\n" % tgt_emb.size())
            for i in range(len(params.tgt_dico)):
                f.write(u"%s %s\n" % (params.tgt_dico[i], " ".join('%.5f' % x for x in tgt_emb[i])))

    # PyTorch file
    if params.export == "pth":
        src_path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.src_lang)
        tgt_path = os.path.join(params.exp_path, 'vectors-%s.pth' % params.tgt_lang)
        logger.info('Writing source embeddings to %s ...' % src_path)
        torch.save({'dico': params.src_dico, 'vectors': src_emb}, src_path)
        logger.info('Writing target embeddings to %s ...' % tgt_path)
        torch.save({'dico': params.tgt_dico, 'vectors': tgt_emb}, tgt_path)

def compute_candidates_for_method_similarity(src_dico,tgt_dico,params):

    # file_path = "dict/candidates_dict.txt"

    src_candidate_indices = list()
    tgt_candidate_indices = list()

    if os.path.isfile(params.identical_dict_path):
        with open(params.identical_dict_path,"r") as f:
            for line in f:
                line = line.replace("\n","")
                splits = line.split("-")
                src_candidate_indices.append(int(splits[0]))
                tgt_candidate_indices.append(int(splits[1]))
    else:
    
        for src_token, src_index  in src_dico.items():
            src_splits = src_token.split(".")
            src_class_method = src_splits[-2:]
            for tgt_token, tgt_index in tgt_dico.items():
                tgt_splits = tgt_token.split(".")
                tgt_class_method = tgt_splits[-2:]

                src_method = src_splits[len(src_splits)-1]
                src_class = src_splits[len(src_splits)-2]

                tgt_method = tgt_splits[len(tgt_splits)-1]
                tgt_class = tgt_splits[len(tgt_splits)-2]


                class_ratio = difflib.SequenceMatcher(None,src_class,tgt_class).ratio()
                method_ratio = difflib.SequenceMatcher(None,src_method,tgt_method).ratio()

                # mean = float((class_ratio + method_ratio)/2)

                # if class_ratio >= 0.8 and method_ratio >= 0.8:


                if method_ratio >= 0.9:
                # if (".".join(src_class_method) == ".".join(tgt_class_method)) or (src_token == tgt_token):
                    print("------------")
                    print(src_token)
                    print(tgt_token, tgt_method)
                    src_candidate_indices.append(src_index)
                    tgt_candidate_indices.append(tgt_index)

                    with open(params.identical_dict_path,"a") as f:
                        f.write(str(src_index) + "-" + str(tgt_index))
                        f.write("\n")


    params.src_candidate_indices = src_candidate_indices
    params.tgt_candidate_indices = tgt_candidate_indices

    return src_candidate_indices, tgt_candidate_indices
