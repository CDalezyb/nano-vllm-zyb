import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from natsort import natsorted


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    model.load_model(path)
    '''  由于有tp, 需要自定义 weight_loader 读取各自的模型权重
    '''
    # packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    # for file in glob(os.path.join(path, "*.safetensors")):  # 遍历 safetensors，结果是键值对的形式
    #     with safe_open(file, "pt", "cpu") as f:
    #         for weight_name in f.keys():
    #             for k in packed_modules_mapping:
    #                 if k in weight_name: # 模型实现跟Qwen开源实现不太一样（名字上），做一些字符串替换,完成开源权重和推理内的代码的映射
    #                     v, shard_id = packed_modules_mapping[k]
    #                     param_name = weight_name.replace(k, v)
    #                     param = model.get_parameter(param_name)
    #                     weight_loader = getattr(param, "weight_loader")
    #                     weight_loader(param, f.get_tensor(weight_name), shard_id)
    #                     break
    #             else:  # for循环如果没有被打断过，就走到else
    #                 param = model.get_parameter(
    #                     weight_name
    #                 )  # 返回对模型某一层权重的“引用”
    #                 # 如果这一层的参数有“weight_loader” 就返回它，否则就使用给定的 default_weight_loader
    #                 weight_loader = getattr(
    #                     param, "weight_loader", default_weight_loader
    #                 )
    #                 weight_loader(param, f.get_tensor(weight_name))


def print_model(path: str):
    if "models" in path:
        display_path = path[path.find("models"):]
    else:
        display_path = path
    print(f"Model Path: {display_path}")
    print("-" * 50)
    
    weight_dict = {}
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                weight_dict[weight_name] = f.get_tensor(weight_name).shape
    
    
    if not weight_dict:
        print("No .safetensors files or weights found in the path.")
        return
    max_name_length = max(len(name) for name in weight_dict.keys())
    
    sorted_weight_names = natsorted(weight_dict.keys())
    for weight_name in sorted_weight_names:
        shape = weight_dict[weight_name]
        print(f"{weight_name.ljust(max_name_length)} | Shape: {shape}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="nano vllm")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the directory containing .safetensors files")
    args = parser.parse_args()
    
    print_model(args.model_path)
