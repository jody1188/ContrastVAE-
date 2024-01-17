import random
import copy
import itertools

import torch


class CombinatorialEnumerate(object):
    """Given M type of augmentations, and a original sequence, successively call \
    the augmentation 2*C(M, 2) times can generate total C(M, 2) augmentaion pairs. 
    In another word, the augmentation method pattern will repeat after every 2*C(M, 2) calls.
    
    For example, M = 3, the argumentation methods to be called are in following order: 
    a1, a2, a1, a3, a2, a3. Which formed three pair-wise augmentations:
    (a1, a2), (a1, a3), (a2, a3) for multi-view contrastive learning.
    """
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2, \
                item_similarity_model=None, mim_model = None, mim_prob = 0.8, mask_id = None, insert_rate=0.3, \
                max_insert_num_per_pos=3, substitute_rate=0.3, n_views=5):

        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma), 
                            Insert(item_similarity_model, insert_rate=insert_rate, 
                                max_insert_num_per_pos=max_insert_num_per_pos),
                            MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id)]
        self.n_views = n_views
        # length of the list == C(M, 2)
        self.augmentation_idx_list = self.__get_augmentation_idx_order()
        self.total_augmentation_samples = len(self.augmentation_idx_list)
        self.cur_augmentation_idx_of_idx = 0     

    def __get_augmentation_idx_order(self):
        augmentation_idx_list = []
        for (view_1, view_2) in itertools.combinations([i for i in range(self.n_views)], 2):
            augmentation_idx_list.append(view_1)
            augmentation_idx_list.append(view_2)
        return augmentation_idx_list

    def __call__(self, sequence):
        augmentation_idx = self.augmentation_idx_list[self.cur_augmentation_idx_of_idx]
        augment_method = self.data_augmentation_methods[augmentation_idx]
        # keep the index of index in range(0, C(M,2))
        self.cur_augmentation_idx_of_idx += 1
        self.cur_augmentation_idx_of_idx = self.cur_augmentation_idx_of_idx % self.total_augmentation_samples
        # print(augment_method.__class__.__name__)
        return augment_method(sequence)


class RandomModelAug(object):
    """Randomly pick one data augmentation type every time call"""
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2):
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta)]
        # print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence):
        #randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods)-1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence)   


class RandomCL4Rec(object):
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2):
        self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta)]

    def __call__(self, sequence):
        #randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods)-1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence)


class Random(object):
    """Randomly pick one data augmentation type every time call"""
    def __init__(self, tao=0.2, gamma=0.7, beta=0.2, \
                item_similarity_model=None, mim_model = None, mim_prob = 0.8, mask_id = None, insert_rate=0.3, \
                max_insert_num_per_pos=3,\
                augment_threshold=-1,
                substitute_rate = 0.1, 
                augment_type_for_short='SIM'):
        self.augment_threshold = augment_threshold
        self.augment_type_for_short = augment_type_for_short
        if self.augment_threshold == -1:
            self.data_augmentation_methods = [MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id)]
            #self.data_augmentation_methods = [Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=beta)]
            print("Total augmentation numbers: ", len(self.data_augmentation_methods))
        elif self.augment_threshold > 0:
            print("short sequence augment type:", self.augment_type_for_short)
            if self.augment_type_for_short == 'SI':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id)]
            elif self.augment_type_for_short == 'SIM':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id),
                                    Mask(gamma=gamma)]

            elif self.augment_type_for_short == 'SIR':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id),
                                    Reorder(beta=gamma)]
            elif self.augment_type_for_short == 'SIC':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id),
                                    Crop(tao=tao)]
            elif self.augment_type_for_short == 'SIMR':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id),
                                    Mask(gamma=gamma), Reorder(beta=gamma)]
            elif self.augment_type_for_short == 'SIMC':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id),
                                    Mask(gamma=gamma), Crop(tao=tao)]
            elif self.augment_type_for_short == 'SIRC':
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id),
                                    Reorder(beta=gamma), Crop(tao=tao)]
            else:
                print("all aug set for short sequences")
                self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                        max_insert_num_per_pos=max_insert_num_per_pos, 
                                        augment_threshold=self.augment_threshold),
                                    MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id),
                                   Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma)]                
            self.long_seq_data_aug_methods = [Insert(item_similarity_model, insert_rate=insert_rate, 
                                    max_insert_num_per_pos=max_insert_num_per_pos, 
                                    augment_threshold=self.augment_threshold),
                                Crop(tao=tao), Mask(gamma=gamma), Reorder(beta=gamma),
                                MIM(mim_model = mim_model, mim_prob = mim_prob, mask_id = mask_id)]
            print("Augmentation methods for Long sequences:", len(self.long_seq_data_aug_methods))
            print("Augmentation methods for short sequences:", len(self.short_seq_data_aug_methods))
        else:
            raise ValueError("Invalid data type.")


    def __call__(self, sequence):
        if self.augment_threshold == -1:
            #randint generate int x in range: a <= x <= b
            augment_method_idx = random.randint(0, len(self.data_augmentation_methods)-1)
            augment_method = self.data_augmentation_methods[augment_method_idx]
            # print(augment_method.__class__.__name__) # debug usage
            return augment_method(sequence)
        elif self.augment_threshold > 0:
            seq_len = len(sequence)
            if seq_len > self.augment_threshold:
                #randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.long_seq_data_aug_methods)-1)
                augment_method = self.long_seq_data_aug_methods[augment_method_idx]
                # print(augment_method.__class__.__name__) # debug usage
                return augment_method(sequence)
            elif seq_len <= self.augment_threshold:
                #randint generate int x in range: a <= x <= b
                augment_method_idx = random.randint(0, len(self.short_seq_data_aug_methods)-1)
                augment_method = self.short_seq_data_aug_methods[augment_method_idx]
                # print(augment_method.__class__.__name__) # debug usage
                return augment_method(sequence)                


def _ensmeble_sim_models(top_k_one, top_k_two):
    # only support top k = 1 case so far
#     print("offline: ",top_k_one, "online: ", top_k_two)
    if top_k_one[0][1] >= top_k_two[0][1]:
        return [top_k_one[0][0]]
    else:
        return [top_k_two[0][0]]


class Insert(object):
    """Insert similar items every time call"""
    def __init__(self, item_similarity_model, insert_rate=0.4, max_insert_num_per_pos=1,
            augment_threshold=14):
        self.augment_threshold = augment_threshold
        if type(item_similarity_model) is list:
            self.item_sim_model_1 = item_similarity_model[0]
            self.item_sim_model_2 = item_similarity_model[1]
            self.ensemble = True
        else:
            self.item_similarity_model = item_similarity_model
            self.ensemble = False
        self.insert_rate = insert_rate
        self.max_insert_num_per_pos = max_insert_num_per_pos
        

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        insert_nums = max(int(self.insert_rate*len(copied_sequence)), 1)
        insert_idx = random.sample([i for i in range(len(copied_sequence))], k = insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos/insert_nums)))
                if self.ensemble:
                    top_k_one = self.item_sim_model_1.most_similar(item,
                                            top_k=top_k, with_score=True)
                    top_k_two = self.item_sim_model_2.most_similar(item,
                                            top_k=top_k, with_score=True)
                    inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two)
                else:
                    inserted_sequence += self.item_similarity_model.most_similar(item,
                                            top_k=top_k)
            inserted_sequence += [item]

        return inserted_sequence


class MIM(object):
    """Substitute with similar items"""
    def __init__(self, mim_model, mim_prob, mask_id):
        self.mask_id = mask_id
        self.model = mim_model
        self.mim_prob = mim_prob

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        sequence = sequence[-50:]
        copied_sequence = copy.deepcopy(torch.Tensor(sequence))
        masked_input_id, labels = mask_tokens(copied_sequence, self.mim_prob, self.mask_id, do_rep_random=False)

        while masked_input_id.cpu().tolist().count(self.mask_id) < 1:
            masked_input_id, labels = mask_tokens(copied_sequence, self.mim_prob, self.mask_id, do_rep_random=False)

        with torch.no_grad():
            masked_input_id = masked_input_id.long().unsqueeze(0)
            labels = labels.long().unsqueeze(0)
            copied_sequence = labels
            predictions = self.model.pretrain(masked_input_id)
        
        copied = copy.deepcopy(masked_input_id.cpu().tolist())
        augmented1 = []

        for sent_idx in range(len(masked_input_id)): # 5 
            copied = copy.deepcopy(copied_sequence.cpu().tolist()[sent_idx])
            for i in range(len(masked_input_id[sent_idx])):
                if masked_input_id[sent_idx][i] == 0: # Padding #
                    break

                if masked_input_id[sent_idx][i] == self.mask_id:
                    org_item = labels.cpu().tolist()[sent_idx][i]
                    prob = predictions[sent_idx][i].softmax(dim=0)
                    probability, candidates = prob.topk(5)
   
                    if probability[0] < 0.95:
                        res = candidate_filtering(copied, i, org_item, candidates)
                    else:
                        res = candidates[0]

                    copied[i] = res
            augmented1.append(copied)

        return augmented1[0]


class Crop(object):
    """Randomly crop a subseq from the original sequence"""
    def __init__(self, tao=0.2):
        self.tao = tao

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.tao*len(copied_sequence))
        #randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1)
        if sub_seq_length<1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index:start_index+sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""
    def __init__(self, gamma=0.7):
        self.gamma = gamma

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(self.gamma*len(copied_sequence))
        mask = [0 for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k = mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""
    def __init__(self, beta=0.2):
        self.beta = beta

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        sub_seq_length = int(self.beta*len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence)-sub_seq_length-1)
        sub_seq = copied_sequence[start_index:start_index+sub_seq_length]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index+sub_seq_length:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


class Identity(object):
    """identity augmentation. Do nothing"""
    def __init__(self):
        # self.beta = beta
        return

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        return copied_sequence


def batch_augment(model, dataset, args):

    k = args.k
    threshold = args.threshold
    mlm_prob = args.mlm_prob
    batch_size = args.pre_batch_size

    pretrain_sampler = RandomSampler(dataset)
    pretrain_dataloader = DataLoader(dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size)

    model.eval()

    augmented_res = []
    
    for batch in tqdm.tqdm(pretrain_dataloader):

        masked_input_id, input_ids = batch[0], batch[1]
        
        labels = input_ids

        with torch.no_grad():
            predictions = model.pretrain(masked_input_id)

        augmented1 = []

        for sent_idx in range(len(masked_input_id)): # 100 
            copied = copy.deepcopy(input_ids.cpu().tolist()[sent_idx])
            for i in range(len(masked_input_id[sent_idx])):
                if masked_input_id[sent_idx][i] == 0: # Padding #
                    break

                if masked_input_id[sent_idx][i] == args.mask_id:
                    org_item = labels.cpu().tolist()[sent_idx][i]
                    prob = predictions[sent_idx][i].softmax(dim=0)
                    probability, candidates = prob.topk(k)
                    if probability[0]<threshold:
                        res = candidate_filtering(copied, i, org_item, candidates)
                    else:
                        res = candidates[0]

                    copied[i] = res

            augmented1.append(copied)
        #########################################################
        augmented_res.extend(augmented1)

    return augmented_res


def mask_tokens(input_ids, mlm_prob, mask_idx, do_rep_random):
   
    labels = input_ids.clone()
    probability_matrix = torch.empty(labels.shape).uniform_(0, 1)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    mask_rep_prob = 0.5
    if not do_rep_random:
        mask_rep_prob = 1.0
    
    indices_replaced = torch.bernoulli(torch.full(labels.shape, mask_rep_prob)).bool() & masked_indices
    input_ids[indices_replaced] = mask_idx

    return input_ids, labels


def candidate_filtering(input_ids, idx, org, candidates):

    candidates_list = candidates.cpu().tolist() 

    for rank, token in enumerate(candidates):
        if org!=token:
            if len(input_ids)-1 == idx:
                 if input_ids[idx-1]==candidates[rank]:
                     continue
            elif input_ids[idx-1]==candidates[rank] or input_ids[idx+1]==candidates_list[rank]:
                continue
            return candidates[rank].item()
    return org