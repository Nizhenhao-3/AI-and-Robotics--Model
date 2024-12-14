import torch
import numpy as np
import zarr
import collections
import os




def create_sample_indices(episode_ends,sequece_length,pad_before,pad_after):
    """
    :param episode_ends:每个episode的结束帧的索引
    :param sequece_length:样本序列的长度
    :return:样本索引的数组
    """
    #sample 和 episode的关系
    #每个episode的sample帧数
    #return的indices每行有4个值，前面俩个值代表sample在全局的开始索引和结束索引，后面两个值表示的是哪些sample中的索引是可以取到buffer的值的
    
    #pad_before obs超出开始帧数，pad_after action超出结束帧数
    indices=[]
    #对episodes进行遍历
    for i in range(len(episode_ends)):
        start_index=0#global
        if i>0:
            start_index=episode_ends[i-1] #相当于第i个episode的开始帧
        end_index=episode_ends[i]#不需要减去1，因为[start_index:end_index]是左闭右开的,最后取到的就是end_index-1 (global)
        episode_length=end_index-start_index

        min_index=-pad_before#pad_before=obs_horizon-1  (local)
        max_index=episode_length-sequece_length+pad_after #(local)

        for j in range(min_index,max_index+1):
            buffer_start_index=max(j,0)+start_index #(global)
            buffer_end_index=min(j+sequece_length,episode_length)+start_index #(global) 

            start_offset=buffer_start_index-(j+start_index) #(local) 
            end_offset=(j+sequece_length+start_index)-buffer_end_index #(local) 超出episode_length的帧数

            sample_start_index=0+start_offset #(local) 
            sample_end_index=sequece_length-end_offset #(local)
            indices.append([buffer_start_index,buffer_end_index,sample_start_index,sample_end_index])
    
    indices=np.array(indices)
    return indices


        #对每个episode进行遍历

def get_data_stats(data):
    """
    获取数据的统计信息（最小值和最大值）
    :param data:数据
    :return:最小值和最大值
    """
    #data的shape为（25650,2），axis=0表示对每一列进行操作，axis=1表示对每一行进行操作
    stats={"min":np.min(data,axis=0),"max":np.max(data,axis=0)} 


        
#数据集包括data和meta，meta包括episode_ends,episode_ends是不同的视频的结束帧，data里包括action,img,keypoint,state
class pushTStateDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):

        dataset_root=zarr.open(dataset_path, mode='r')
        train_data={'action':dataset_root['data']['action'][:],'obs':dataset_root['data']['state'][:]}#action.shape(25650,2),obs.shape(25650,5)
        episode_ends=dataset_root['meta']['episode_ends'][:]#numpy数据，大小为（206，）
        #计算每个状态-动作序列的开始和结束索引
        indices=create_sample_indices(episode_ends=episode_ends,
                                      sequece_length=pred_horizon,
                                      pad_before=obs_horizon-1,
                                      pad_after=action_horizon-1)
        #计算统计信息并将数据归一化到【-1，1】
        stats={}
        normalized_train_data={}
        for key,data in train_data.items():#key是action和obs,data是action和obs对应的数据
            
            stats[key]=get_data_stats(data)
            stats[key]['std']=np.std(train_data[key],axis=0)
            normalized_train_data[key]=(train_data[key]-stats[key]['mean'])/stats

        

    def __len__(self):
        return self.dataset_length
    def __getitem__(self, idx):
        data = self.dataset[idx]
        obs = data['obs']
        action = data['action']
        next_obs = data['next_obs']
        return obs, action, next_obs


def main():
    pred_horizon = 16
    obs_horizon = 2
    action_horizon=8

    dataset_path='./diffusion_policy_learn/pusht_cchi_v7_replay.zarr.zip'

    if not os.path.exists(dataset_path):
        print(f"dataset_path {dataset_path} does not exist")
        return
    
    dataset=pushTStateDataset(dataset_path, pred_horizon, obs_horizon, action_horizon)

       

if __name__ == '__main__':
    main()