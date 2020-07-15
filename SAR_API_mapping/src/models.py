import torch
from torch import nn

from .utils import load_embeddings, normalize_embeddings


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        # 词向量的维度
        self.emb_dim = params.emb_dim
        # 判别器的层数 default 2
        self.dis_layers = params.dis_layers
        # 判别器中间层的维度，是2048
        self.dis_hid_dim = params.dis_hid_dim
        # 神经网络的dropout，是为了防止模型的过拟合， default 0
        self.dis_dropout = params.dis_dropout
        # 输入的dropout，default为0.1
        self.dis_input_dropout = params.dis_input_dropout

        layers = [nn.Dropout(self.dis_input_dropout)]
        for i in range(self.dis_layers + 1):
            # 第一层的输入维度是 emb_dim，因为第一层的输入是词向量。其余层的输入维度是2048
            input_dim = self.emb_dim if i == 0 else self.dis_hid_dim
            # 最后一层的输出维度是1，因为判别器的输出就是一个概率。其余层的输出维度是2048
            output_dim = 1 if i == self.dis_layers else self.dis_hid_dim
            layers.append(nn.Linear(input_dim, output_dim))
            if i < self.dis_layers:
                layers.append(nn.LeakyReLU(0.2))
                layers.append(nn.Dropout(self.dis_dropout))
        #  这一步是将输出转换成0-1的概率
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # 此处的x就是判别器的输入。要确保输入的向量的维度是二
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        return self.layers(x).view(-1)


def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    # 生成src_dico的词向量,生成一个大矩阵，len(src_dico)表示的是单词数，后面的参数表示的是生成的词向量的维度
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    # 刚刚生成的src_emb只是一个矩阵，是没有数据的，这一步将之前生成的API的词向量拷贝到src_emb中
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    # mapping
    # 对输入的矩阵做线性变换,初始化adversarial training的輸入矩陣
    mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=False)
    if getattr(params, 'map_id_init', True):
        mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
        mapping.cuda()
        if with_dis:
            discriminator.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping, discriminator
