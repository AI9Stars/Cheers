
import json
from torch.utils.data import Dataset
import os
from PIL import Image

class DpgDataset(Dataset):
    def __init__(self, dir_file_path="./Evaluation_DPGBench/benchmarks/dpg/prompts"):
        self.dir_file_path = dir_file_path
        self.prompt = []
        self.prompt_files_name = os.listdir(dir_file_path)
        self.total_metadatas = len(self.prompt_files_name)
        self.gen_num = 4
        self.benchmark_type = "gen"

    def __len__(self):
        return self.total_metadatas

    def __getitem__(self, idx):
        file_path = os.path.join(self.dir_file_path, self.prompt_files_name[idx])
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt = file.read()
        return prompt

    def gen_num_images(self):
        return self.gen_num
    
    def output_form(self, idx, work_dir, image_list):
        file_name = self.prompt_files_name[idx].split(".")[0]
        os.makedirs(work_dir, exist_ok=True)
        # breakpoint()
        assert len(image_list) == self.gen_num 
        width, height = image_list[0].size
        new_image = Image.new('RGB', (width * 2, height * 2))
        new_image.paste(image_list[0], (0,0))
        new_image.paste(image_list[1], (width,0))
        new_image.paste(image_list[2], (0,height))
        new_image.paste(image_list[3], (width,height))
        new_image.save(os.path.join(work_dir, file_name + ".jpg"))




    def eval_result(self, data_work_dir):
        pass

    def datatype(self):
        return self.benchmark_type
        