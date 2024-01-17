import os
import pickle

from tqdm import tqdm

import torch
import torch.nn as nn
from modules import Encoder, LayerNorm, Decoder, VariationalDropout
import math
import numpy as np
import random

def symmetric(X):

    return X.triu() + X.triu(1).transpose(-1, -2)

class EASE(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.rand(self.hidden_size , self.hidden_size).fill_diagonal_(0))
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        A = symmetric(self.weight)

        return self.sigmoid(x @ A)

class ContrastVAE(nn.Module):

    def __init__(self, args):
        super(ContrastVAE, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder_mu = Encoder(args)
        self.item_encoder_logvar = Encoder(args)
        self.item_encoder_context = Encoder(args)
        self.item_decoder = Decoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.r1 = torch.ones_like(torch.Tensor(args.batch_size,args.max_seq_length,1)).cuda()
        self.args = args
        self.latent_dropout = nn.Dropout(args.reparam_dropout_rate)
        self.apply(self.init_weights)
        self.temperature = nn.Parameter(torch.zeros(1), requires_grad=True)

    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence) # shape: b*max_Sq*d
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb # shape: b*max_Sq*d


    def extended_attention_mask(self, input_ids):
        attention_mask = (input_ids > 0).long()# used for mu, var
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64 b*1*1*max_Sq
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8 for causality
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1) #1*1*max_Sq*max_Sq
        subsequent_mask = subsequent_mask.long()

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask #shape: b*1*max_Sq*max_Sq
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask


    def eps_anneal_function(self, step):

        return min(1.0, (1.0*step)/self.args.total_annealing_step)

    def reparameterization(self, mu, logvar, step):  # vanila reparam

        std = torch.exp(0.5 * logvar)
        if self.training:
            eps = torch.randn_like(std)
            res = mu + std * eps
        else: res = mu + std
        return res

    def reparameterization1(self, mu, logvar, step): # reparam without noise
        std = torch.exp(0.5*logvar)
        return mu+std


    def reparameterization2(self, mu, logvar, step): # use dropout

        if self.training:
            std = self.latent_dropout(torch.exp(0.5*logvar))
        else: std = torch.exp(0.5*logvar)
        res = mu + std
        return res

    def reparameterization3(self, mu, logvar,step): # apply classical dropout on whole result
        std = torch.exp(0.5*logvar)
        res = self.latent_dropout(mu + std)
        return res

    def reparameterization4(self,mu,logvar,step):
        std = torch.exp(0.5*logvar)
        if self.training:
            try:
                eps_ = dirichlet.Dirichlet(self.r1) # 256 X 50 X 1
                eps = eps_.sample().cuda() # 256 X 50 X 1
                res = mu + std*eps
            except:
                res= mu+std
        else: 
            res = mu + std
        return res


    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def encode(self, sequence_emb, extended_attention_mask,sequence_emb_var,extended_attention_mask_var,sequence_context,extended_attention_mask_context): # forward

        item_encoded_mu_layers = self.item_encoder_mu(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True)

        item_encoded_logvar_layers = self.item_encoder_logvar(sequence_emb_var, extended_attention_mask_var,
                                                              True)
        context_encoded_logvar_layers = self.item_encoder_context(sequence_context, extended_attention_mask_context,
                                                              True)                                                    

        return item_encoded_mu_layers[-1], item_encoded_logvar_layers[-1], context_encoded_logvar_layers[-1]

    def decode(self, z, extended_attention_mask,context = None):
        item_decoder_layers = self.item_decoder(z,
                                                extended_attention_mask,
                                                output_all_encoded_layers = True,context = context )
        sequence_output = item_decoder_layers[-1]
        return sequence_output

    def forward(self, input_ids, step):

        sequence_emb = self.add_position_embedding(input_ids)# shape: b*max_Sq*d
        sequence_emb_var = self.add_position_embedding(input_ids)
        sequence_context = self.add_position_embedding(input_ids)
        extended_attention_mask = self.extended_attention_mask(input_ids)
        extended_attention_mask_var = self.extended_attention_mask(input_ids)
        extended_attention_mask_context = self.extended_attention_mask(input_ids)
        sequence_ease = self.add_position_embedding(input_ids)

        if self.args.latent_contrastive_learning:
            mu1, log_var1,context1 = self.encode(sequence_emb, extended_attention_mask,sequence_emb_var,extended_attention_mask_var,sequence_context,extended_attention_mask_context)
            mu2, log_var2,context2= self.encode(sequence_emb, extended_attention_mask,sequence_emb_var,extended_attention_mask_var,sequence_context,extended_attention_mask_context)
            m = nn.Softmax(dim=1)

            if self.args.diri:
                try:
                    self.r1 += m(reconstructed_seq1.sum(dim=-1)).unsqueeze(-1) # 256 50 1
                    self.r1 = (self.r1-self.r1.min())/(self.r1.max() - self.r1.min())
                except:
                    self.r1 = self.r1
                    print('Exception!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                z1 = self.reparameterization4(mu1, log_var1, step)
            else:
                z1 = self.reparameterization4(mu1, log_var1, step)

            z2 = self.reparameterization2(mu2, log_var2, step)

            if self.args.context:
                reconstructed_seq1 = self.decode(z1, extended_attention_mask,context = z1)
                reconstructed_seq1 = reconstructed_seq1 + z1
                reconstructed_seq2 = self.decode(z2, extended_attention_mask,context = z2)
                reconstructed_seq2 = reconstructed_seq2 + z2
            else:
                reconstructed_seq1 = self.decode(z1, extended_attention_mask,context = None)
                reconstructed_seq2 = self.decode(z2, extended_attention_mask,context = None)
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        elif self.args.latent_data_augmentation:
            aug_sequence_emb = self.add_position_embedding(aug_input_ids)  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(aug_input_ids)
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2 = self.reparameterization2(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        elif self.args.Joint_VAE:

            mu1, log_var1,context1 = self.encode(sequence_emb, extended_attention_mask,sequence_emb_var,extended_attention_mask_var,sequence_context,extended_attention_mask_context)
            mu2, log_var2,context2= self.encode(sequence_emb, extended_attention_mask,sequence_emb_var,extended_attention_mask_var,sequence_context,extended_attention_mask_context)
            m = nn.Softmax(dim=1)

            if self.args.diri:
                try:
                    self.r1 += m(reconstructed_seq1.sum(dim=-1)).unsqueeze(-1) # 256 50 1
                    self.r1 = (self.r1-self.r1.min())/(self.r1.max() - self.r1.min())
                except:
                    self.r1 = self.r1
                    print('Exception!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                z1 = self.reparameterization4(mu1, log_var1, step)
            else:
                z1 = self.reparameterization1(mu1, log_var1, step)

            z2 = self.reparameterization4(mu2, log_var2, step)

            if self.args.context:
                reconstructed_seq1 = self.decode(z1, extended_attention_mask,context = context1)
                reconstructed_seq1 = reconstructed_seq1 + z1
                reconstructed_seq2 = self.decode(z2, extended_attention_mask,context = context2)
                reconstructed_seq2 = reconstructed_seq2 + z1
            else:
                reconstructed_seq1 = self.decode(z1, extended_attention_mask,context = None)
                reconstructed_seq2 = self.decode(z2, extended_attention_mask,context = None)
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2

        elif self.args.latent_data_augmentation:
            aug_sequence_emb = self.add_position_embedding(aug_input_ids)  # shape: b*max_Sq*d
            aug_extended_attention_mask = self.extended_attention_mask(aug_input_ids)
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2 = self.reparameterization2(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)
            
            return reconstructed_seq1, reconstructed_seq2, mu1, mu2, log_var1, log_var2, z1, z2



class ContrastVAE_VD(ContrastVAE):

    def __init__(self, args):
        super(ContrastVAE, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.item_encoder_mu = Encoder(args)
        self.item_encoder_logvar = Encoder(args)
        self.item_decoder = Decoder(args)

        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.latent_dropout_VD = VariationalDropout(inputshape=[args.max_seq_length, args.hidden_size], adaptive='layerwise')
        self.latent_dropout = nn.Dropout(0.1)
        self.args = args
        self.apply(self.init_weights)

        self.drop_rate = nn.Parameter(torch.tensor(0.2), requires_grad=True)


    def reparameterization3(self, mu, logvar, step): # use drop out

        std, alpha = self.latent_dropout_VD(torch.exp(0.5*logvar))
        res = mu + std
        return res, alpha


    def pool(self, x):
        # Shape of x - (layer_count, batch_size, seq_length, hidden_size)
        x = torch.stack(x[1:])
        x = x.transpose(0, 1)
        if self.pooling_strategy == "mean":
            return x[:, -1, :, :].mean(dim=1)
        elif self.pooling_strategy == "max":
            return torch.max(x[:, -1, :, :], dim=1)[0]  # Pool from last layer.
        else:
            raise Exception("Wrong pooling strategy!")


    def forward(self, input_ids, step):
        if self.args.variational_dropout:
            sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
            extended_attention_mask = self.extended_attention_mask(input_ids)
            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            mu2, log_var2 = self.encode(sequence_emb, extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)
            z2, alpha = self.reparameterization3(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            reconstructed_seq2 = self.decode(z2, extended_attention_mask)

        elif self.args.Joint_VAE:
            sequence_emb = self.add_position_embedding(input_ids)  # shape: b*max_Sq*d
            extended_attention_mask = self.extended_attention_mask(input_ids)
            #aug_sequence_emb = self.add_position_embedding(augmented_input_ids)  # shape: b*max_Sq*d
            #aug_extended_attention_mask = self.extended_attention_mask(augmented_input_ids)

            mu1, log_var1 = self.encode(sequence_emb, extended_attention_mask)
            #mu2, log_var2 = self.encode(aug_sequence_emb, aug_extended_attention_mask)
            z1 = self.reparameterization1(mu1, log_var1, step)

            #z2, alpha = self.reparameterization3(mu2, log_var2, step)
            reconstructed_seq1 = self.decode(z1, extended_attention_mask)
            #reconstructed_seq2 = self.decode(z2, extended_attention_mask)


        return reconstructed_seq1, mu1, log_var1, z1



class OnlineItemSimilarity:

    def __init__(self, item_size):
        self.item_size = item_size
        self.item_embeddings = None
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.total_item_list = torch.tensor([i for i in range(self.item_size)],
                                            dtype=torch.long).to(self.device)
        # self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        
    def update_embedding_matrix(self, item_embeddings):
        print(item_embeddings)
        self.item_embeddings = copy.deepcopy(item_embeddings)
        self.base_embedding_matrix =self.item_embeddings(self.total_item_list)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embeddings(torch.tensor(item_idx).to(self.device)).view(-1, 1)
                item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                print("ssssss")
                continue
        return max_score, min_score
        
    def most_similar(self, item_idx, top_k=1, with_score=False):
        

        item_idx = torch.tensor(item_idx, dtype=torch.long).to(self.device)
        item_vector = self.item_embeddings(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (item_similarity - self.min_score) / (self.max_score - self.min_score)
        #remove item idx itself
        values, indices = item_similarity.topk(top_k+1)
        if with_score:
            item_list = indices.tolist()
            score_list = values.tolist()
            if item_idx in item_list:
                idd = item_list.index(item_idx)
                item_list.remove(item_idx)
                score_list.pop(idd)
            return list(zip(item_list, score_list))
        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list

class OfflineItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF', \
        dataset_name='Sports_and_Outdoors'):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()
        
    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score
    
    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user,item,record in data:
            train_data_dict.setdefault(user,{})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path = './similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid,itemid,int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self,train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        N.setdefault(i,0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i,{})
                            C[i].setdefault(j,0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i,0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i,{})
                            C[i].setdefault(j,0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item,{})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item,0);
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'Item2Vec':
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            print("Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(sentences=self.train_data_list,
                                        vector_size=20, window=5, min_count=0, 
                                        epochs=100)
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=20)
                self.itemSimBest.setdefault(cur_item,{})
                for (related_item, score) in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item,0)
                    self.itemSimBest[cur_item][related_item] = score
            print("Item2Vec model saved to: ", save_path)
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'LightGCN':
            # train a item embedding from lightGCN model, and then convert to sim dict
            print("generating similarity model..")
            itemSimBest = light_gcn.generate_similarity_from_light_gcn(self.dataset_name)
            print("LightGCN based model saved to: ", save_path)
            self._save_dict(itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(),key=lambda x : x[1], \
                                            reverse=True)[0:top_k]
                if with_score:
                    return list(map(lambda x: (int(x[0]), (float(x[1]) - self.min_score)/(self.max_score -self.min_score)), top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(),key=lambda x : x[1], \
                                            reverse=True)[0:top_k]
                if with_score:
                    return list(map(lambda x: (int(x[0]), (float(x[1]) - self.min_score)/(self.max_score -self.min_score)), top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k = top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))



class S3RecModel(nn.Module):
    def __init__(self, args):
        super(S3RecModel, self).__init__()
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args = args

        # add unique dense layer for 4 losses respectively
        self.mip_norm = nn.Linear(args.hidden_size, args.item_size)
        self.softmax = nn.LogSoftmax(dim = -1)
        #self.criterion = nn.BCELoss(reduction='none')
        self.apply(self.init_weights)


    def masked_item_prediction(self, sequence_output):
        '''
        :param sequence_output: [B L H]
        :param target_item: [B L H]
        :return: scores [B*L]
        '''
        sequence_output = self.mip_norm(sequence_output) # [B*L H]
        
        #target_item = target_item.view([-1,self.args.hidden_size]) # [B*L H]
        #score = torch.mul(sequence_output, target_item) # [B*L H]
        #return torch.sigmoid(torch.sum(score, -1)) # [B*L]
        return sequence_output


    def add_position_embedding(self, sequence):

        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        item_embeddings = self.item_embeddings(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    def pretrain(self, masked_item_sequence):

        # Encode masked sequence
        sequence_emb = self.add_position_embedding(masked_item_sequence)
        sequence_mask = (masked_item_sequence == 0).float() * -1e8
        sequence_mask = torch.unsqueeze(torch.unsqueeze(sequence_mask, 1), 1)

        encoded_layers = self.item_encoder(sequence_emb,
                                          sequence_mask,
                                          output_all_encoded_layers=True)
        # [B L H]
        sequence_output = encoded_layers[-1]

        # MIP
        #item_embs = self.item_embeddings(items)
        mim_output = self.masked_item_prediction(sequence_output)
        #mip_loss = self.criterion(mip_score, torch.ones_like(torch.sigmoid(mip_score), dtype=torch.float32))
        #mip_mask = (masked_item_sequence == self.args.mask_id).float()
        #mip_loss = torch.sum(mip_loss * mip_mask.flatten())

        return mim_output

    def init_weights(self, module):

        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()