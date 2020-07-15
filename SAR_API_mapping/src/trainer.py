# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
from logging import getLogger
import scipy
import scipy.linalg
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from .utils import get_optimizer, load_embeddings, normalize_embeddings, export_embeddings
from .utils import clip_parameters
from .dico_builder import build_dictionary
from .evaluation.word_translation import DIC_EVAL_PATH, load_identical_char_dico, load_dictionary


logger = getLogger()


class Trainer(object):

    def __init__(self, src_emb, tgt_emb, mapping, discriminator, params):
        """
        Initialize trainer script.
        """
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.src_dico = params.src_dico
        self.tgt_dico = getattr(params, 'tgt_dico', None)
        self.mapping = mapping
        self.discriminator = discriminator
        self.params = params
        
        # optimizers
        if hasattr(params, 'map_optimizer'):
            optim_fn, optim_params = get_optimizer(params.map_optimizer)
            self.map_optimizer = optim_fn(mapping.parameters(), **optim_params)
        if hasattr(params, 'dis_optimizer'):
            optim_fn, optim_params = get_optimizer(params.dis_optimizer)
            self.dis_optimizer = optim_fn(discriminator.parameters(), **optim_params)
        else:
            assert discriminator is None

        # best validation score
        self.best_valid_metric = -1e12

        self.decrease_lr = False

    def get_dis_xy(self, volatile):
        """
        Get discriminator input batch / output target.
        """
        # select random word IDs
        bs = self.params.batch_size
        # 选K个最常见的单词来进行判别，故K要小于src词典的大小和tgt词典的大小
        mf = self.params.dis_most_frequent
        assert mf <= min(len(self.src_dico), len(self.tgt_dico))
        # 在0到词典长度-1（或mf - 1）间随机取值初始化，其实对应过来就是字典的id
        with torch.no_grad():
            src_ids = torch.zeros(bs, dtype=torch.long).random_(len(self.src_dico) if mf == 0 else mf)
            tgt_ids = torch.zeros(bs, dtype=torch.long).random_(len(self.tgt_dico) if mf == 0 else mf)
        if self.params.cuda:
            src_ids = src_ids.cuda()
            tgt_ids = tgt_ids.cuda()

        # get word embeddings
        # 取出id在src_ids中的词向量 将Variable(src_ids, volatile=True)改为src_ids
        with torch.no_grad():
            src_emb = self.src_emb(src_ids)
            tgt_emb = self.tgt_emb(tgt_ids)
        if volatile:
            with torch.no_grad():
                src_emb = self.mapping(src_emb.detach())
                tgt_emb = tgt_emb.detach()
        else :
            src_emb = self.mapping(src_emb.detach())
            tgt_emb = tgt_emb.detach()

        x = torch.cat([src_emb, tgt_emb], 0)
        y = torch.zeros(2 * bs)
        # 使用的是标签平滑正则化。dis_smooth默认值0.1
        y[:bs] =  self.params.dis_smooth
        y[bs:] = 1 - self.params.dis_smooth
        y = (y.cuda() if self.params.cuda else y)

        return x, y

    def dis_step(self, stats):
        """
        Train the discriminator.
        """
        self.dis_optimizer.zero_grad()

        self.discriminator.train()
        bs = self.params.batch_size

        # loss
        x, y = self.get_dis_xy(volatile=True)

        fake = x[:bs]
        real = x[bs:]
        # Variable(x.data)
        #  Train real
        preds_real = self.discriminator(real)
        preds_real = preds_real.mean(0).view(1)
        preds_real.backward(torch.tensor([1], dtype=torch.float))

        # Train fake
        preds_fake = self.discriminator(fake)
        preds_fake = preds_fake.mean(0).view(1)
        preds_fake.backward(torch.tensor([1], dtype=torch.float) * -1)
        # 计算交叉熵。preds是预测值，y是target
        loss = preds_fake - preds_real
        Wasserstein_D = preds_real - preds_fake
        # loss = (F.binary_cross_entropy(preds_fake, y)) / 1
        # 将loss.data[0]转换成loss.item()
        stats['DIS_COSTS'].append(loss.item())

        # check NaN
        # (loss != loss).data.any()
        if torch.isnan(loss):
            logger.error("NaN detected (discriminator)")
            exit()

        # optim
        #
        # loss.backward()
        self.dis_optimizer.step()
        clip_parameters(self.discriminator, self.params.dis_clip_weights)

    def mapping_step(self, stats):
        """
        Fooling discriminator training step.
        """
        if self.params.dis_lambda == 0:
            return 0

        self.map_optimizer.zero_grad()
        self.discriminator.eval()
        bs = self.params.batch_size

        # loss
        x, y = self.get_dis_xy(volatile=False)
        fake = x[:bs]
        preds = self.discriminator(fake)
        loss = preds.mean().mean(0).view(1)
        loss.backward(torch.tensor([1], dtype=torch.float))
        stats['Gen_COSTS'].append(loss.item())
        # loss = self.params.dis_lambda * loss

        # check NaN
        # (loss != loss).data.any()
        if torch.isnan(loss):
            logger.error("NaN detected (fool discriminator)")
            exit()

        # optim

        # loss.backward()
        self.map_optimizer.step()
        self.orthogonalize()

        return 2 * self.params.batch_size

    def load_training_dico(self, dico_train):
        """
        Load training dictionary.
        """
        word2id1 = self.src_dico.word2id
        word2id2 = self.tgt_dico.word2id

        # identical character strings
        if dico_train == "identical_char":
            self.dico = load_identical_char_dico(word2id1, word2id2)
        # use one of the provided dictionary
        elif dico_train == "default":
            filename = '%s-%s.0-5000.txt' % (self.params.src_lang, self.params.tgt_lang)
            self.dico = load_dictionary(
                os.path.join(DIC_EVAL_PATH, filename),
                word2id1, word2id2
            )
        # dictionary provided by the user
        else:
            self.dico = load_dictionary(dico_train, word2id1, word2id2)

        # cuda
        if self.params.cuda:
            self.dico = self.dico.cuda()

    def build_dictionary(self, mutual_nn=0):
        """
        Build a dictionary from aligned embeddings.
        """
        src_emb = self.mapping(self.src_emb.weight).data
        tgt_emb = self.tgt_emb.weight.data
        src_emb = src_emb / src_emb.norm(2, 1, keepdim=True).expand_as(src_emb)
        tgt_emb = tgt_emb / tgt_emb.norm(2, 1, keepdim=True).expand_as(tgt_emb)
        self.dico = build_dictionary(src_emb, tgt_emb, mutual_nn, self.params)

    def procrustes(self):
        """
        Find the best orthogonal matrix mapping using the Orthogonal Procrustes problem
        https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem
        """
        self_dico = self.dico[:, 0]
        A = self.src_emb.weight.data[self.dico[:, 0]]
        B = self.tgt_emb.weight.data[self.dico[:, 1]]
        W = self.mapping.weight.data
        M = B.transpose(0, 1).mm(A).cpu().numpy()
        U, S, V_t = scipy.linalg.svd(M, full_matrices=True)
        W.copy_(torch.from_numpy(U.dot(V_t)).type_as(W))

    def orthogonalize(self):
        """
        Orthogonalize the mapping.
        """
        if self.params.map_beta > 0:
            W = self.mapping.weight.data
            beta = self.params.map_beta
            W.copy_((1 + beta) * W - beta * W.mm(W.transpose(0, 1).mm(W)))

    def update_lr(self, to_log, metric):
        """
        Update learning rate when using SGD.
        """
        if 'sgd' not in self.params.map_optimizer:
            return
        old_lr = self.map_optimizer.param_groups[0]['lr']
        new_lr = max(self.params.min_lr, old_lr * self.params.lr_decay)
        if new_lr < old_lr:
            logger.info("Decreasing learning rate: %.8f -> %.8f" % (old_lr, new_lr))
            self.map_optimizer.param_groups[0]['lr'] = new_lr

        if self.params.lr_shrink < 1:
        # if self.params.lr_shrink < 1 and to_log[metric] >= -1e7:
            # if to_log[metric] < self.best_valid_metric:
                # logger.info("Validation metric is smaller than the best: %.5f vs %.5f"
                #             % (to_log[metric], self.best_valid_metric))
                # decrease the learning rate, only if this is the
                # second time the validation metric decreases
            if self.decrease_lr:
                old_lr = self.map_optimizer.param_groups[0]['lr']
                self.map_optimizer.param_groups[0]['lr'] *= self.params.lr_shrink
                logger.info("Shrinking the learning rate: %.5f -> %.5f"
                            % (old_lr, self.map_optimizer.param_groups[0]['lr']))
            self.decrease_lr = True

    def save_best(self, to_log, metric):
        """
        Save the best model for the given validation metric.
        """
        # best mapping for the given validation criterion
        # print(to_log)
        # if to_log[metric] > self.best_valid_metric:
        #     # new best mapping
        #     self.best_valid_metric = to_log[metric]
        #     logger.info('* Best value for "%s": %.5f' % (metric, to_log[metric]))
            # save the mapping
        W = self.mapping.weight.data.cpu().numpy()
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        # logger.info('* Saving the mapping to %s ...' % path)
        torch.save(W, path)

    def reload_best(self):
        """
        Reload the best mapping.
        """
        path = os.path.join(self.params.exp_path, 'best_mapping.pth')
        logger.info('* Reloading the best model from %s ...' % path)
        # reload the model
        assert os.path.isfile(path)
        to_reload = torch.from_numpy(torch.load(path))
        W = self.mapping.weight.data
        assert to_reload.size() == W.size()
        W.copy_(to_reload.type_as(W))

    def export(self):
        """
        Export embeddings.
        """
        params = self.params

        # load all embeddings
        logger.info("Reloading all embeddings for mapping ...")
        params.src_dico, src_emb = load_embeddings(params, source=True, full_vocab=True)
        params.tgt_dico, tgt_emb = load_embeddings(params, source=False, full_vocab=True)

        # apply same normalization as during training
        normalize_embeddings(src_emb, params.normalize_embeddings, mean=params.src_mean)
        normalize_embeddings(tgt_emb, params.normalize_embeddings, mean=params.tgt_mean)

        # map source embeddings to the target space
        bs = 4096
        logger.info("Map source embeddings to the target space ...")
        for i, k in enumerate(range(0, len(src_emb), bs)):
            with torch.no_grad():
                x = (src_emb[k:k + bs])
            src_emb[k:k + bs] = self.mapping(x.cuda() if params.cuda else x).data.cpu()

        # write embeddings to the disk
        export_embeddings(src_emb, tgt_emb, params)
