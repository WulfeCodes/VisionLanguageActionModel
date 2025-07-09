import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.ops import roi_align
from detectron2.checkpoint import DetectionCheckpointer
import numpy as np
from numpy import load
import glob
from pathlib import Path
from PIL import Image
import subprocess
import boto3
import io
import torchvision
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from botocore.exceptions import ClientError
from torchvision import transforms
from transformers import RobertaTokenizer, RobertaModel

# If you want higher‑level S3 transfer helpers
from boto3.s3.transfer import S3Transfer
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util

# Some basic setup:
# Setup detectron2 logger
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn

#TODO MaskRCNN training: train on aws,
#TODO NEXT STEPS: CREATE INFERENCE FUNCTION FOR ITERATIVE GENERATIONS, CREATE TRAINING, think about encoder training, BPE/DCT
#incorporate beam search inference?
#check for start/padding/end tokens in CALVIN, 
# import some common libraries
import os, json, cv2, random
# import some common detectron2 utilities
#understand the embedding logic + 

class Encoder(nn.Module):
    def __init__(self,
                 max_length,
                 src_vocab_size,
                 embed_size,
                #  mask,
                 num_length, 
                 heads, 
                 device, 
                 final_expansion,dropout):
        super(Encoder,self).__init__()
        self.positional_encoding = nn.Embedding(max_length,embed_size)
        self.device = device
        self.word_encoding = nn.Linear(768,embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_size,heads,final_expansion,dropout)
                for _ in range(num_length)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        #TODO consider sinusoidal dynamic embedding size
        x = x.squeeze(0)
        mask = mask.squeeze(0)
        N, seq_length = x.shape
        x=self.word_encoding(x)
        x=x.long()
        positions = torch.arange(N, device=self.device)
        positions = positions.long()
        out=self.dropout(x + self.positional_encoding(positions)) 
        out = out.unsqueeze(0)
        for layer in self.layers:
            out = layer(out, out, out, mask)
        out = out.mean(dim=1,keepdim=True)
        return out

class Decoder(nn.Module):
    def __init__(self,trg_vocab_size,max_size, embed_size, dropout, heads,
                 forward_expansion,device,encSize,num_layers):
        super(Decoder,self).__init__()

        self.positional_embeddings = nn.Embedding(max_size,embed_size)
        self.token_embeddings = nn.Linear(7,embed_size)
        self.layers=nn.ModuleList(
            DecoderBlock(forward_expansion,dropout,embed_size,heads,encSize,device)
            for _ in range(num_layers)
        )
        self.device = device
        self.fc_out = nn.Linear(embed_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x,trg,enc_out,trg_mask,src_mask):

        B,N, seq_length = trg.shape
        positions = torch.arange(N, device=self.device)
        positions = positions.long()
        out=self.dropout((self.token_embeddings(trg)).long() + self.positional_embeddings(positions)) 

        for layer in self.layers:
            #took out src_mask 
            out = layer(out,enc_out,enc_out,src_mask,trg_mask)
        out = self.fc_out(out)
        return out

class DecoderBlock(nn.Module):
    def __init__(self,forward_expansion,dropout,embed_size,heads,encSize,device):
        super(DecoderBlock,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)
        self.attention = selfAttention(embed_size,heads)
        self.transfurmer = TransformerBlock(embed_size,heads,forward_expansion,dropout)
        self.keyProjection = nn.Linear(encSize,embed_size)
        self.valueProjection = nn.Linear(encSize,embed_size)  

        self.encSize = encSize
    def forward(self,x,key,value,src_mask,trg_mask):

        attention = self.attention.forward(x,x,x,trg_mask)
        #attend mask properly, feed thru check, logic shape
        y=self.dropout(self.layer_norm(attention+x))
        if len(key[-1]) != len(x[-1]):
            value=self.valueProjection(value)
            key = self.keyProjection(key)

        out=self.transfurmer.forward(value,key,y,trg_mask)
        return out

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size, #400 for max natural language command
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            max_src_size,
            max_trg_size,
            embed_size=768,
            enc_emb_size=384,
            num_layers=6,
            forward_expansion=4,
            heads = 10,
            dropout = 0,
            device="cpu",
            max_length = 5000,
            actionDim = 7

    ):
        super(Transformer,self).__init__()
        self.embedding_size=embed_size
        self.max_src_size = max_src_size
        self.max_trg_size = max_trg_size
        
        self.encoder = Encoder(
            src_vocab_size=768,
            max_length=300,
            embed_size=enc_emb_size,
            num_length=6,
            heads=8,
  # Parameter 'forward_expansion' from Transformer passed to Encoder's 'forward_expansion'
           # Parameter 'dropout' from Transformer passed to Encoder's 'dropout'
            dropout=0, 
            final_expansion=forward_expansion,
            device="cpu", 
        )

        self.decoder = Decoder(
            trg_vocab_size,
            max_size=100,
            embed_size=768,
            num_layers=6,
            heads=8,
            forward_expansion=forward_expansion,
            dropout=0,
            encSize=enc_emb_size,
            device='cpu',
        )
    
        self.src_pad_idx = src_pad_idx
        self.src_trg_pad_idx = trg_pad_idx
        self.device = 'cpu'
        #TODO add correct ignore index
        #tokenization inits 
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.token_model = RobertaModel.from_pretrained('roberta-base')
        self.embedding_size = self.token_model.config.hidden_size
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids('</s>')
        self.pretrained_sep_emb = self.token_model.embeddings.word_embeddings.weight[self.sep_token_id].unsqueeze(0).unsqueeze(0)  # shape (1,1,D)
        self.sep_token = nn.Parameter(self.pretrained_sep_emb.clone())  # Initialize with pretrained
        self.actionProj = nn.Linear(trg_vocab_size,actionDim)
        self.imageProj = nn.Linear(1,self.embedding_size)


    def MultiModalProjection(self,raw_text,raw_img):
        tokenized_txt=self.tokenizer(raw_text,return_tensors='pt',padding=True, truncation=True)
        print(f"keys:{tokenized_txt.keys()}")
        #tokenized_txt['attention_mask']
        text_output = self.token_model(**tokenized_txt)
        cat_txt = text_output.last_hidden_state  # unpacks input_ids, attention_mask
        cat_img = raw_img.unsqueeze(-1)
        img_proj = self.imageProj(cat_img)
        print(img_proj.shape,cat_txt.shape)
        x = torch.cat([cat_txt,self.sep_token,img_proj],dim=1)
        return x

    def make_src_mask(self,src):
        padded = torch.zeros(len(src), len(src[0]), self.embedding_size)
        #padded is actual data with 0s being non vals
        #maks is a flattened tensor with Trues being non vals
        mask = torch.ones(len(src),len(src[0]),dtype=torch.bool)

        for i, seq in enumerate(src):
            length = seq.size(0)
            padded[i,:length,:] = seq
            mask[i,:length]=False
        mask = mask.unsqueeze(0).unsqueeze(1).unsqueeze(2)  # [1, 1, 1, 300]

        return mask.to(device)

    
    def make_trg_mask(self,trg):
        B, T, _ = trg.shape  # Batch size, target length, feature dim

        # Create lower triangular mask
        causal_mask = torch.tril(torch.ones((T, T), device=trg.device)).bool()  # (T, T)

        # Expand to (B, 1, T, T) for broadcasting across heads
        causal_mask = causal_mask.unsqueeze(0).expand(B, 1, T, T)

        return causal_mask.to(device)
    
    def inference(self,src,max_len=100,start_token=1, end_token=2):
        #used for inference mode 
        self.eval()

        #not sure if no grad is neccesary here
        with torch.no_grad():
            batch_size = src.size(0)
            src_mask = self.make_src_mask(src)
            enc_src = self.encoder(src,src_mask)
            #initialize target sequence
            trg = torch.full((batch_size,1),start_token, dtype=torch.long, device=self.device)

            for _ in range(max_len - 1):
                trg_mask = self.make_trg_mask(trg)
                out = self.decoder(trg,enc_src,src_mask=src_mask,trg_mask=trg_mask)

                next_token_logits = out[:,-1,:]
                #TODO analyze out's shapes and cooresponding probs
                next_token = torch.argmax(next_token_logits, dim=-1,keepdim=True)
                trg = torch.cat([trg,next_token],dim=1)

                if(next_token == end_token).all():
                    break
                return trg
        

    def forward(self,src,trg,enc_op=None):
        if enc_op == None:
            src_mask = self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg)
            enc_src = self.encoder(src,src_mask)
            out = self.decoder(src,trg, enc_src, src_mask=src_mask,trg_mask=trg_mask)
            
            out = self.actionProj(out)
            #should we squeeze now?
            return out[:,-1,:],enc_op
        else: 
            src_mask = self.make_src_mask(src)
            trg_mask = self.make_trg_mask(trg)
            enc_src = self.encoder(src,src_mask)
            out = self.decoder(src,trg, enc_src, src_mask=src_mask,trg_mask=trg_mask)
            
            out = self.actionProj(out)
            #should we squeeze now?
            return out[:,-1,:]
        #decoder arguments: self,x,enc_out,trg_mask,src_mask)
    
    def compute_loss(self,output,trg):
        logits = output.view(-1,output.size(-1))
        targets = targets.view(-1)

        loss = self.criterion(logits,targets)
        return loss




class TransformerBlock(nn.Module):
    def __init__(self,embedding_size,heads,expansion_rate,dropout):
        super(TransformerBlock,self).__init__()
        self.embedding_size = embedding_size
        self.heads = heads
        self.expansion_rate = expansion_rate
        self.Attention = selfAttention(self.embedding_size,self.heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(self.embedding_size,self.embedding_size*expansion_rate),
            nn.ReLU(),
            nn.Linear(self.embedding_size*expansion_rate,self.embedding_size)
        )

        self.l1 = nn.LayerNorm(embedding_size)
        self.l2 = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,value,key,query,mask,srcType=None):

        attention = self.Attention.forward(value, key, query, mask)
        x = self.dropout(self.l1(attention+query))
        y = self.feed_forward(x)
        out = self.dropout(self.l2(x+y))
        
        return out
        
        

        #use dropout after the linear and norm
        

class selfAttention(nn.Module):
    def __init__(self,embedding_size,heads):
        super(selfAttention,self).__init__()
        self.embedding_size = embedding_size
        self.heads = heads
        #embedding size%head num must equal 0!

        self.head_size = (embedding_size // heads)
        assert (self.head_size * heads == embedding_size)
        self.values = nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.queries = nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.keys = nn.Linear(self.embedding_size,self.embedding_size,bias=False)
        self.fc_out = nn.Linear(self.embedding_size,self.embedding_size,bias=False)

    def forward(self,values,keys,queries,mask,srcType=None):

        value_len, keys_len, queries_len = values.shape[1], keys.shape[1], queries.shape[1]
        N = queries.shape[0]

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        values=values.reshape(N, value_len, self.heads, self.head_size)
        queries=queries.reshape(N, queries_len, self.heads, self.head_size)
        keys=keys.reshape(N, keys_len, self.heads, self.head_size)

        energy = torch.einsum("nqhd,nkhd->nhqk",[queries,keys])


        if mask is not None:
            energy.masked_fill(mask == 0,float(1e-24))
        attention = f.softmax(energy/((self.embedding_size)**(1/2)),dim=3)
        out = torch.einsum("nhql,nlhd->nqhd",[attention,values]).reshape(N,queries_len,self.embedding_size)
        out = self.fc_out(out)
        return out

def get_aws(keyy):
    session = boto3.Session(region_name = "us-east-2")
    s3_client = session.client('s3',region_name="us-east-2")
    buffer=io.BytesIO()
    bucket = 'vla-bucket1'
    resp = s3_client.get_object(Bucket=bucket,Key=keyy)
    data=resp['Body'].read()
    loaded = np.load(io.BytesIO(data),allow_pickle=True)
    print(f"successfully loaded {loaded}")
        #*GULP*#
    # sagemaker = boto3.client('sagemaker')
    # sagemaker.create_training_job(TrainingJobName="my-detectron2-job",
    #         AlgorithmSpecification={
    #             'TrainingImage': '763104351884.dkr.ecr.us-east-2.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker',
    #             'TrainingInputMode': 'File'},
    #         RoleArn='arn:aws:iam::346717860071:role/SageMakerTrainer',
    #         InputDataConfig = [{
    #             'ChannelName': 'training',
    #             'DataSource': {
    #                 'S3DataSource': {
    #                     'S3DataType': 'S3Prefix',
    #                     'S3Uri': 's3://vla-bucket1/semantic/',
    #                     'S3DataDistributionType': 'FullyReplicated'
    #                 }
    #             }
    #         }],
    #                     OutputDataConfig = {'S3OutputPath':'s3://vla-bucket1/semantic-model-artifacts/'},
    #         ResourceConfig={'InstanceType': 'ml.m5.xlarge','InstanceCount': 1,'VolumeSizeInGB': 30},                          
    #         StoppingCondition={'MaxRuntimeInSeconds': 1800}                              
    #                               )  
    
    return loaded

def load_aws(list,keyy):
    d_type = [('path', 'U500'), ('array', 'O')]

    np_arr=np.array(list,dtype=d_type)
    session = boto3.Session(region_name = "us-east-2")
    s3_client = session.client('s3',region_name="us-east-2")
    buffer=io.BytesIO()
    bucket = 'vla-bucket1'
    np.save(buffer,np_arr,allow_pickle=True)
    buffer.seek(0)

    s3_client.put_object(Bucket=bucket,Key=keyy,Body=buffer.getvalue())
    print("successfully loaded dataset")


class Dataset:
    def __init__(self,image_dir="C:/Users/vijay/Downloads/data"):

        self.image_dir = image_dir
        self.train_dataset = None#self.load_data()
        self.val_dataset = None
        self.total_dataset = self.load_data()

    def load_valData(self):
        return None
        return self.total_dataset[:6500]

    def load_trainData(self):
        return None
        return self.total_dataset[6500:]

    def load_data(self):
        #Temporary!
        return None
        pairs = set()
        Dataset = []

        for i, folder in enumerate(os.listdir(self.image_dir)):

            curr_folder_path=os.path.join(self.image_dir,folder)

            for j, folder_ in enumerate(os.listdir(curr_folder_path)):

                # if os.path.isdir(os.path.join(curr_folder_path,folder_)):
                sub_folder=Path(os.path.join(curr_folder_path,folder_))

                all_pngs = list(sub_folder.glob("*.png"))
                file_names = {f.name.strip() for f in all_pngs}
                file_paths = {f.name: f for f in all_pngs}

                for fname in file_names:
                    if fname.startswith("mask_color_"):
                        counterpart = fname.replace("mask_color","img",1)
                        if counterpart in file_names:
                            pairs.add((file_paths[counterpart],file_paths[fname]))

                    elif fname.startswith("img"): 
                        mask_name = fname.replace("img","mask_color")
                        if mask_name in file_names:
                            pairs.add((file_paths[fname], file_paths[mask_name]))

        if len(pairs)==0:
            print("storage error")

        else:
            print("pairs length",len(pairs))

            for z,CurrTuple in enumerate((pairs)):
                if CurrTuple[1].name.startswith("mask_color_"):
                    CurrTuple_1=self.create_binary_mask(CurrTuple[1])

                    Dataset.append((CurrTuple[0],CurrTuple_1))
                else:
                    print(f"Unexpected pair: {CurrTuple[0].name} -> {CurrTuple[1].name}")
                    #CurrTuple[1] is now a H,W np binary mask
        print("finished populating sets",len(Dataset))

        #if aws is not used simply return Dataset
        d_type = [('path', 'U500'), ('array', 'O')]

        np_arr=np.array(Dataset,dtype=d_type)
        
        correct_dataset=self.fine_tune(np_arr)
        #TODO add data augmentation
        return correct_dataset
    def create_binary_mask(self,img_path,target_color=(251, 154, 153)):
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        # Create binary mask: 1 where pixel matches target_color, else 0
        mask = np.all(img_np == target_color, axis=-1).astype(np.uint8)

        return mask  # shape: (H, W), dtype: uint8 (values: 0 or 1)
    
    def mask_to_polygon(self,mask_tensor):
        mask_np = mask_tensor.cpu().numpy().astype('uint8')
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            if len(contour) >= 3:  # Need at least 3 points
                polygon = contour.flatten().tolist()
                if len(polygon) >= 6:  # Need at least 3 coordinate pairs
                    polygons.append(polygon)
        return polygons


    def crop_tight(self,bbox,mask,img):

        y_min,y_max = int(bbox[0][1].item()),int(bbox[0][3].item())
        x_min,x_max = int(bbox[0][0].item()),int(bbox[0][2].item())
        imgR = img[:,y_min:y_max, x_min:x_max]  # assuming CHW format
        shaper = mask[y_min:y_max,x_min:x_max]
        maskR = torch.from_numpy(shaper)

        print("cropped")

        return maskR,imgR

    def add_augmented_data(self, target_list, augment_ratio=1):
        augmented_list = target_list.copy()  # keep originals
    
        for imgT, maskT in target_list:
            for _ in range(augment_ratio):
                # Generate random parameters for each augmentation
                transform = A.Compose([
                    A.HorizontalFlip(p=random.random()),
                    A.Rotate(limit=random.randint(5, 30), p=random.random()),
                    A.RandomBrightnessContrast(
                        brightness_limit=random.uniform(0.1, 0.3),
                        contrast_limit=random.uniform(0.1, 0.3),
                        p=random.random()
                    ),
                    A.Blur(blur_limit=random.choice([1, 3, 5, 7, 9]), p=random.random()),
                ])
                
                # Convert and apply augmentation
                img_np = imgT.permute(1, 2, 0).numpy()
                mask_np = maskT.numpy()
                
                augmented = transform(image=img_np, mask=mask_np)
                
                aug_img = torch.from_numpy(augmented['image']).permute(2, 0, 1)
                aug_mask = torch.from_numpy(augmented['mask'])
                
                augmented_list.append([aug_img, aug_mask])
        
        return augmented_list


    def fine_tune(self,dataset): 
        target_list = []
        samples = []
        #TODO editing of training needs to occur for loss computation
        #current binary approach moves non robot classifications toward a background distribution
        for i,(img,binary_mask) in enumerate(dataset):

            tmp0 = torch.from_numpy(binary_mask)
            #torch.where creates a tuple of coords

            if tmp0.sum() == 0:
                print(f"Skipping frame {i} - no robot arm detected")
                continue
            pos = torch.where(tmp0)

            boxes = torch.tensor([[pos[1].min(), pos[0].min(),
                                pos[1].max(), pos[0].max()]], dtype=torch.float32)

            img1 = Image.open(img)
            transform = transforms.ToTensor()
            imgT = transform(img1)
            maskT = torch.from_numpy(binary_mask)
            target_list.append([imgT,maskT])
        print("converted data")

        augmented_data = self.add_augmented_data(target_list)
        print("formatted original list of data dicts",len(augmented_data))

        for i, (imgT, maskT) in enumerate(augmented_data):
            # Get bounding box from mask

            if maskT.sum() == 0:
                print(f"Skipping frame {i} - no robot arm detected")
                continue

            # Get image dimensions
            img_path = "C:/Project/DL_HW/dataset/processed_images/image{i}.jpg"
            img_pil = Image.fromarray(imgT.permute(1,2,0).numpy().astype('uint8'))
            img_pil = img_pil.resize((256,256),Image.LANCZOS)
            mask_pil = Image.fromarray(maskT.numpy().astype('uint8') * 255)
            mask_resized = mask_pil.resize((256, 256), Image.NEAREST)
            maskT = torch.tensor(np.array(mask_resized) > 0)
            pos = torch.where(maskT > 0)

            if pos[0].numel() == 0 or pos[1].numel() == 0:
                print(f"Skipping frame {i} - no nonzero pixels in resized mask")
                continue

            boxes = torch.tensor([[pos[1].min(), pos[0].min(),
                                pos[1].max(), pos[0].max()]], dtype=torch.float32)        
            width, height =img_pil.size
            img_pil.save(img_path)
            # Create sample dict
            sample = {
                "file_name": img_path,
                "height": height, 
                "width": width,
                "image_id": i,
                "annotations": [{
                    "bbox": boxes[0].tolist(),
                    "category_id": 80,  # robot_arm class
                    "segmentation": self.mask_to_polygon(maskT),
                    "area": maskT.sum().item(),
                    "iscrowd": 0,
                    "bbox_mode": BoxMode.XYXY_ABS
                }]
            }
            samples.append(sample)
        print("finished data to detectron2 formaat")

        return samples

class Semanatic_Model():

    def __init__(self,weight_directory=None,weight_file=None,Dataset_obj=None,device='cpu'):
        self.weight_dir = weight_directory
        self.weight_file = weight_file
        self.weight_path = (f'{self.weight_dir}/{weight_file}')
        self.Dataset_obj = Dataset_obj
        self.dataset = None#Dataset_obj.load_data()
        self.cfg = get_cfg()
        self.device = device   
        self.model,self.optimizer=self.init_model()
        self.backbone = self.model.backbone
        self.fine_tuned = False
    def init_model(self):

        DatasetCatalog.register("robosemantic_dataset",self.Dataset_obj.load_trainData)
        DatasetCatalog.register("robot_val",self.Dataset_obj.load_valData)
        if self.weight_path==None:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        else:
            self.cfg.MODEL.WEIGHTS = self.weight_path
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

        # Dataset configuration
        self.cfg.DATASETS.TRAIN = ("robosemantic_dataset",)
        self.cfg.DATASETS.TEST = ("robot_val",)
        self.cfg.DATALOADER.NUM_WORKERS = 0
        
        # Model configuration - CRITICAL PART FOR CUSTOM CLASSES
        # COCO has 80 classes, we're adding 1 more (robot_arm) = 81 total
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = 81  # 80 COCO + 1 robot class

        self.cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
        self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
        self.cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 1.0
        self.cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
        self.cfg.TEST.EVAL_PERIOD = 0
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 1000
        self.cfg.SOLVER.IMS_PER_BATCH = 1
        self.cfg.SOLVER.BASE_LR = 0.00025
        self.cfg.SOLVER.MAX_ITER = 6000  # Adjust based on your dataset size
        self.cfg.SOLVER.STEPS = []         # Learning rate decay steps

        #TODO determine how batch training is going to work w/ aws and data augmentation
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2
        self.cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
        if torch.cuda.is_available():
            self.cfg.MODEL.DEVICE = "cuda"
        else:
            self.cfg.MODEL.DEVICE = "cpu"
        # Output directory
        self.cfg.OUTPUT_DIR = self.weight_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        model=build_model(self.cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        return model, optimizer

    def save_model(self):
        try:
            checkpointer=DetectionCheckpointer(self.model,save_dir=self.weight_dir)
            checkpointer.save(file=self.weight_file)
            self.fine_tuned = True
            print("successfully saved")
        except Exception as e:
            print("error in saving model")

    def load_model(self,type,weight_path):
        if type == 'train':
            if self.fine_tuned:
                self.cfg.MODEL.WEIGHTS =weight_path
                trainer = DefaultTrainer(self.cfg)
                trainer.resume_or_load(resume=False)
                trainer.train()
            else:
                trainer = DefaultTrainer(self.cfg)
                trainer.resume_or_load(resume=False)
                trainer.train()
            return trainer
        elif type == 'test':
            if self.fine_tuned:
                self.cfg.MODEL.WEIGHTS = (weight_path)
                predictor = DefaultPredictor(self.cfg)
            else:
                predictor = DefaultPredictor(self.cfg)
            return predictor

    def predict(self,image):
        image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard ImageNet normalization
                        std=[0.229, 0.224, 0.225])
        ])

        img_pil = transform(image).unsqueeze(0)
        features=self.backbone(img_pil)
        feature_highres=features['p2']
        pooled = f.adaptive_avg_pool2d(feature_highres, (1, 1))
        flattened = pooled.flatten(1)  #1,256 shape
        return flattened
    def fine_tune(self):
        trainer = DefaultTrainer(self.cfg)
        trainer.train()

        print("dataset0 training finished")
        try:
            self.save_model()
            print("dataset successfully saved")
            self.fine_tuned = True
        except Exception as e:
            print(f"saving failed with {e}")         
        #check if there are internal folders if so parse through those
            #sort mask_color_* to their counterparts

    #model.eval()
#    features: ['action', 'observation.state', 'timestamp', 'frame_index', 'episode_index', 'index', 'task_index'],

if __name__ == '__main__':

    print("hello")
    device = torch.device('cpu')

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 300
    trg_vocab_size = 384 #trg may be too big here, but better safe than sorry
    data = load('C:/Project/DL_HW/dataset/calvin_debug_dataset/training/episode_0358656.npz')
    lst = data.files
    data1 = load('C:/Project/DL_HW/dataset/calvin_debug_dataset/training/lang_annotations/auto_lang_ann.npy',allow_pickle=True)
    structured_object = data1[()]

    # #files return all the keys of the npz dictionary?
    # print("annotated",data1)
    # print("potated",data2)
    #TODO what are the contents of ep_lens.npy and ep_start_ends_ids.npy
    ep_lens = load('C:/Project/DL_HW/dataset/calvin_debug_dataset/training/lang_annotations/auto_lang_ann.npy',allow_pickle=True)
    ep_lens_obj = ep_lens[()]

    if isinstance(ep_lens_obj, dict):
        keys = ep_lens_obj.keys()
        print(f"The keys in the structured object are: {list(keys)}")
        keys0 = ep_lens_obj['language'].keys()
        keys1 = ep_lens_obj['info'].keys()
        print(f"the keys0,1 in the eps len bject are {keys0},{keys1}")
        print(ep_lens_obj['language']['task'][0])
        print("length",len(ep_lens_obj['language']['task']))
        #ep_lens_obj['info']['episodes'] is blank
        print(ep_lens_obj['info']['indx'][0])

        #length 9
        #TODO figure out which camera view of roboseg aligns with which rgb calvin
        #ep_start_ends_ids_obj['info']['episodes'] are blank
        #TASK KEYS ARE JOINT EMBEDDING LANGUAGE INPUT: ep_lens_obj['language']['ann'][i]
        print("number of task inputs:",len(ep_lens_obj['language']['ann']))
        print("number of encoding outputs:",len(ep_lens_obj['language']['emb']))
        print("length of available tasks: ",len(ep_lens_obj['info']['indx']))
        #Contains short pseudo language notation for tasks language input: ep_lens_obj['language']['task'][0]
        #EMB are hierarchal latent encoder output: ep_lens_obj['language']['emb'][i]
        #Contains indexes of start to end frames ep_lens_obj['info']['indx'][i]
        #ep_start_ends_ids_obj['info']['indx'][i] contains raw shapes of cameras and grippers
    for item in lst:
        print(item)
        print(data[item].shape)
    print("successfully loaded CALVIN info!")
    dataset_obj =Dataset()
    #Dataset=get_aws('semantic/dataset.npy')

    #dataset is a list of length 2 tuples where [0] is orig img and [1] is a np binary mask for class pred 92 (robot arm)
    print("successful save and load")
    sem_model = Semanatic_Model(weight_directory='./cv_models',weight_file="model_final.pth",Dataset_obj=dataset_obj)
    sem_model.fine_tuned = True

    print("inited model")
     # Built from config

    print("saved model")

    #TODO Later model load test

    saved_path="./output/model_robot_final.pth"
    
    print("loaded model test")
    #TODO FINISH FINE TUNE METHOD overnight
    #sem_model.fine_tune()
    print("fine tuning complete")

    transfurmer= Transformer(src_vocab_size,trg_vocab_size,src_pad_idx,trg_pad_idx,max_src_size=300,max_trg_size=300,actionDim=7).to(
        device
    )
    
    #TODO Logic for simulated action loop" 
    print("number of task inputs:",len(ep_lens_obj['language']['ann']))
    print("number of encoding outputs:",len(ep_lens_obj['language']['emb']))
    print("length of available tasks: ",len(ep_lens_obj['info']['indx']))
    #len(ep_lens_obj['language']['ann'])
    
    for i in range(1): #len(ep_lens_obj['info']['indx'])
        print("episode ranges")
        start_index, end_index =(ep_lens_obj['info']['indx'][i])
        print(start_index,end_index)
        episode_range = end_index-start_index

        start_path=f"C:/Project/DL_HW/dataset/calvin_debug_dataset/training/episode_0{start_index}.npz"
        end_path = f"C:/Project/DL_HW/dataset/calvin_debug_dataset/training/episode_0{end_index}.npz"
        start_data = load(start_path)
        end_data=load(end_path)
        start_files = start_data.files
        concatenated_op=torch.FloatTensor(1,episode_range+1,7)
        curr_vision_ip = start_data['rgb_static']
        curr_emb_op=ep_lens_obj['language']['emb'][i]
        curr_language_ip=ep_lens_obj['language']['ann'][i]

        for j in range(episode_range+1):
            curr_index=start_index+j
            curr_path = f"C:/Project/DL_HW/dataset/calvin_debug_dataset/training/episode_0{curr_index}.npz"
            curr_file=load(curr_path)
            curr_keys = curr_file.files
            concatenated_op[0,j]=torch.from_numpy(curr_file['rel_actions'])

            if j == 0:
                firstImg=curr_file['rgb_static']
                semantic_inference=sem_model.predict(firstImg)
                x=transfurmer.MultiModalProjection(raw_text=curr_language_ip,raw_img=semantic_inference)
                trg_output,enc_output=transfurmer.forward(x,concatenated_op)
            else:
                output=transfurmer.forward(x,concatenated_op,enc_output)    
            #compute loss of end
            #will use rel actions for learning
        
        print(curr_language_ip)
        print(concatenated_op.shape)
        #img need to be a 255x255

        
        #bound to be error prone
        #TODO full pass w encoder and decoder output


    print("model successfully initialized")
    #out = model(x,trg[:,:-1])
    #model.compute_loss(out,trg)
    #print(out.shape)