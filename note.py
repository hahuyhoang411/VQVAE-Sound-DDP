# import math, pickle, os
# import numpy as np
# import torch
# from torch.autograd import Variable
# from torch import optim
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from utils import *
# import sys
# import models.nocond as nc
# import models.vqvae as vqvae
# import models.wavernn1 as wr
# import utils.env as env
# import argparse
# import platform
# import re
# import utils.logger as logger
# import time
# import subprocess
# import torch.multiprocessing as mp
# import config

# import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

# def setup():
#     dist.init_process_group("nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# def cleanup():
#     dist.destroy_process_group()

# def main():
#     setup()
#     parser = argparse.ArgumentParser(description='Train or run some neural net')
#     parser.add_argument('--generate', '-g', action='store_true')
#     parser.add_argument('--float', action='store_true')
#     parser.add_argument('--half', action='store_true')
#     parser.add_argument('--load', '-l')
#     parser.add_argument('--scratch', action='store_true')
#     parser.add_argument('--model', '-m')
#     parser.add_argument('--force', action='store_true', help='skip the version check')
#     parser.add_argument('--count', '-c', type=int, default=3, help='size of the test set')
#     parser.add_argument('--partial', action='append', default=[], help='model to partially load')
#     args = parser.parse_args()

#     if args.float and args.half:
#         sys.exit('--float and --half cannot be specified together')
    
#     if args.float:
#         use_half = False
#     elif args.half:
#         use_half = True
#     else:
#         use_half = False
    
#     model_type = args.model or 'vqvae'
    
#     model_name = f'{model_type}.43.upconv'

#     if model_type == 'vqvae':
#         model_fn = lambda train_set: vqvae.Model(rnn_dims=896, fc_dims=896, global_decoder_cond_dims=train_set.num_speakers(),
#                       upsample_factors=(4, 4, 4), normalize_vq=True, noise_x=True, noise_y=True).cuda()
#         dataset_type = 'multi'
#     elif model_type == 'wavernn':
#         model_fn = lambda dataset: wr.Model(rnn_dims=896, fc_dims=896, pad=2,
#                       upsample_factors=(4, 4, 4), feat_dims=80).cuda()
#         dataset_type = 'single'
#     elif model_type == 'nc':
#         model_fn = lambda dataset: nc.Model(rnn_dims=896, fc_dims=896).cuda()
#         dataset_type = 'single'
#     else:
#         sys.exit(f'Unknown model: {model_type}')

#     if dataset_type == 'multi':
#         data_path = config.multi_speaker_data_path
#         with open(f'{data_path}/index.pkl', 'rb') as f:
#             index = pickle.load(f)
#         test_index = [x[-1:] if i < 2 * args.count else [] for i, x in enumerate(index)]
#         train_index = [x[:-1] if i < args.count else x for i, x in enumerate(index)]
#         train_set = env.MultispeakerDataset(train_index, data_path)
#         print("Number of speakers:", train_set.num_speakers())
#         dataset = DataLoader(train_set, batch_size=256, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_set), num_workers=16)
#     elif dataset_type == 'single':
#         data_path = config.single_speaker_data_path
#         with open(f'{data_path}/dataset_ids.pkl', 'rb') as f:
#             index = pickle.load(f)
#         test_index = index[-args.count:] + index[:args.count]
#         train_index = index[:-args.count]
#         dataset = env.AudiobookDataset(train_index, data_path)
#     else:
#         raise RuntimeError('bad dataset type')
    
#     print(f'dataset size: {len(dataset)}')

#     # Wrap the model with DDP
#     model = model_fn(train_set).to(rank)
#     model = DDP(model, device_ids=[rank])

#     if use_half:
#         model = model.half()

#     for partial_path in args.partial:
#         model.load_state_dict(torch.load(partial_path), strict=False)
    
#     paths = env.Paths(model_name, data_path)
    
#     if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
#         # Start from scratch
#         step = 0
#     else:
#         if args.load:
#             prev_model_name = re.sub(r'_[0-9]+$', '', re.sub(r'\.pyt$', '', os.path.basename(args.load)))
#             prev_model_basename = prev_model_name.split('_')[0]
#             model_basename = model_name.split('_')[0]
#             if prev_model_basename != model_basename and not args.force:
#                 sys.exit(f'refusing to load {args.load} because its basename ({prev_model_basename}) is not {model_basename}')
#             if args.generate:
#                 paths = env.Paths(prev_model_name, data_path)
#             prev_path = args.load
#         else:
#             prev_path = paths.model_path()
#         step = env.restore(prev_path, model)
    
#     #model.freeze_encoder()
    
#     optimiser = optim.AdamW(model.parameters())
    
#     if args.generate:
#         model.do_generate(paths, step, data_path, test_index, use_half=use_half, verbose=True)#, deterministic=True)
#     else:
#         logger.set_logfile(paths.logfile_path())
#         logger.log('------------------------------------------------------------')
#         logger.log('-- New training session starts here ------------------------')
#         logger.log(time.strftime('%c UTC', time.gmtime()))
#         model.do_train(paths, dataset, optimiser, epochs=1000, batch_size=256, step=step, lr=1e-4, use_half=use_half, valid_index=test_index)
        
#     cleanup()

# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)


### EXAMPLE Pytorch

# import torch
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader

# import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
# import os
# import torch
# from torch.utils.data import Dataset

# class MyTrainDataset(Dataset):
#     def __init__(self, size):
#         self.size = size
#         self.data = [(torch.rand(20), torch.rand(1)) for _ in range(size)]

#     def __len__(self):
#         return self.size
    
#     def __getitem__(self, index):
#         return self.data[index]

# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# class Trainer:
#     def __init__(
#         self,
#         model: torch.nn.Module,
#         train_data: DataLoader,
#         optimizer: torch.optim.Optimizer,
#         save_every: int,
#         snapshot_path: str,
#     ) -> None:
#         self.gpu_id = int(os.environ["LOCAL_RANK"])
#         self.model = model.to(self.gpu_id)
#         self.train_data = train_data
#         self.optimizer = optimizer
#         self.save_every = save_every
#         self.epochs_run = 0
#         self.snapshot_path = snapshot_path
#         if os.path.exists(snapshot_path):
#             print("Loading snapshot")
#             self._load_snapshot(snapshot_path)

#         self.model = DDP(self.model, device_ids=[self.gpu_id])

#     def _load_snapshot(self, snapshot_path):
#         loc = f"cuda:{self.gpu_id}"
#         snapshot = torch.load(snapshot_path, map_location=loc)
#         self.model.load_state_dict(snapshot["MODEL_STATE"])
#         self.epochs_run = snapshot["EPOCHS_RUN"]
#         print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

#     def _run_batch(self, source, targets):
#         self.optimizer.zero_grad()
#         output = self.model(source)
#         loss = F.cross_entropy(output, targets)
#         loss.backward()
#         self.optimizer.step()

#     def _run_epoch(self, epoch):
#         b_sz = len(next(iter(self.train_data))[0])
#         print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
#         self.train_data.sampler.set_epoch(epoch)
#         for source, targets in self.train_data:
#             source = source.to(self.gpu_id)
#             targets = targets.to(self.gpu_id)
#             self._run_batch(source, targets)

#     def _save_snapshot(self, epoch):
#         snapshot = {
#             "MODEL_STATE": self.model.module.state_dict(),
#             "EPOCHS_RUN": epoch,
#         }
#         torch.save(snapshot, self.snapshot_path)
#         print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

#     def train(self, max_epochs: int):
#         for epoch in range(self.epochs_run, max_epochs):
#             self._run_epoch(epoch)
#             if self.gpu_id == 0 and epoch % self.save_every == 0:
#                 self._save_snapshot(epoch)


# def load_train_objs():
#     train_set = MyTrainDataset(2048)  # load your dataset
#     model = torch.nn.Linear(20, 1)  # load your model
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     return train_set, model, optimizer


# def prepare_dataloader(dataset: Dataset, batch_size: int):
#     return DataLoader(
#         dataset,
#         batch_size=batch_size,
#         pin_memory=True,
#         shuffle=False,
#         sampler=DistributedSampler(dataset)
#     )


# def main(save_every: int, total_epochs: int, batch_size: int, snapshot_path: str = "snapshot.pt"):
#     ddp_setup()
#     dataset, model, optimizer = load_train_objs()
#     train_data = prepare_dataloader(dataset, batch_size)
#     trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
#     trainer.train(total_epochs)
#     destroy_process_group()


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='simple distributed training job')
#     parser.add_argument('total_epochs', default=10, type=int, help='Total epochs to train the model')
#     parser.add_argument('save_every', default=500, type=int, help='How often to save a snapshot')
#     parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
#     args = parser.parse_args()
    
#     main(args.save_every, args.total_epochs, args.batch_size)


### SINGLE GPU
# import math, pickle, os
# import numpy as np
# import torch
# from torch.autograd import Variable
# from torch import optim
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from utils import *
# import sys
# import models.nocond as nc
# import models.vqvae as vqvae
# import models.wavernn1 as wr
# import utils.env as env
# import argparse
# import platform
# import re
# import utils.logger as logger
# import time
# import subprocess

# import config

# parser = argparse.ArgumentParser(description='Train or run some neural net')
# parser.add_argument('--generate', '-g', action='store_true')
# parser.add_argument('--float', action='store_true')
# parser.add_argument('--half', action='store_true')
# parser.add_argument('--load', '-l')
# parser.add_argument('--scratch', action='store_true')
# parser.add_argument('--model', '-m')
# parser.add_argument('--force', action='store_true', help='skip the version check')
# parser.add_argument('--count', '-c', type=int, default=3, help='size of the test set')
# parser.add_argument('--partial', action='append', default=[], help='model to partially load')
# args = parser.parse_args()

# if args.float and args.half:
#     sys.exit('--float and --half cannot be specified together')

# if args.float:
#     use_half = False
# elif args.half:
#     use_half = True
# else:
#     use_half = False

# model_type = args.model or 'vqvae'

# model_name = f'{model_type}.43.upconv'

# if model_type == 'vqvae':
#     model_fn = lambda dataset: vqvae.Model(rnn_dims=896, fc_dims=896, global_decoder_cond_dims=dataset.num_speakers(),
#                   upsample_factors=(4, 4, 4), normalize_vq=True, noise_x=True, noise_y=True).cuda()
#     dataset_type = 'multi'
# else:
#     sys.exit(f'Unknown model: {model_type}')

# if dataset_type == 'multi':
#     data_path = config.multi_speaker_data_path
#     with open(f'{data_path}/index.pkl', 'rb') as f:
#         index = pickle.load(f)
#     test_index = [x[-1:] if i < 2 * args.count else [] for i, x in enumerate(index)]
#     train_index = [x[:-1] if i < args.count else x for i, x in enumerate(index)]
#     dataset = env.MultispeakerDataset(train_index, data_path)
# else:
#     raise RuntimeError('bad dataset type')

# print(f'dataset size: {len(dataset)}')

# model = model_fn(dataset)

# if use_half:
#     model = model.half()

# for partial_path in args.partial:
#     model.load_state_dict(torch.load(partial_path), strict=False)

# paths = env.Paths(model_name, data_path)

# if args.scratch or args.load == None and not os.path.exists(paths.model_path()):
#     # Start from scratch
#     step = 0
# else:
#     if args.load:
#         prev_model_name = re.sub(r'_[0-9]+$', '', re.sub(r'\.pyt$', '', os.path.basename(args.load)))
#         prev_model_basename = prev_model_name.split('_')[0]
#         model_basename = model_name.split('_')[0]
#         if prev_model_basename != model_basename and not args.force:
#             sys.exit(f'refusing to load {args.load} because its basename ({prev_model_basename}) is not {model_basename}')
#         if args.generate:
#             paths = env.Paths(prev_model_name, data_path)
#         prev_path = args.load
#     else:
#         prev_path = paths.model_path()
#     step = env.restore(prev_path, model)

# #model.freeze_encoder()

# optimiser = optim.AdamW(model.parameters())

# if args.generate:
#     model.do_generate(paths, step, data_path, test_index, use_half=use_half, verbose=True)#, deterministic=True)
# else:
#     logger.set_logfile(paths.logfile_path())
#     logger.log('------------------------------------------------------------')
#     logger.log('-- New training session starts here ------------------------')
#     logger.log(time.strftime('%c UTC', time.gmtime()))
#     model.do_train(paths, dataset, optimiser, epochs=1000, batch_size=256, step=step, lr=1e-4, use_half=use_half, valid_index=test_index)

### EXAMPLE

# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.distributed import init_process_group, destroy_process_group
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler

# import numpy as np
# import pickle
# import config
# import models.vqvae as vqvae
# import utils.env as env
# import utils.logger as logger

# # Dataset Class
# class MultispeakerDataset(Dataset):
#     def __init__(self, index, path):
#         self.path = path
#         self.index = index
#         self.all_files = [(i, name) for (i, speaker) in enumerate(index) for name in speaker]

#     def __getitem__(self, index):
#         speaker_id, name = self.all_files[index]
#         speaker_onehot = (np.arange(len(self.index)) == speaker_id).astype(np.int_)
#         audio = np.load(f'{self.path}/{speaker_id}/{name}.npy')
#         return speaker_onehot, audio

#     def __len__(self):
#         return len(self.all_files)

#     def num_speakers(self):
#         return len(self.index)

# # Setup for DDP
# def ddp_setup():
#     init_process_group(backend="nccl")
#     torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

# class Trainer:
#     def __init__(self, model, train_data, optimizer, save_every, snapshot_path):
#         self.gpu_id = int(os.environ["LOCAL_RANK"])
#         self.model = model.to(self.gpu_id)
#         self.train_data = train_data
#         self.optimizer = optimizer
#         self.save_every = save_every
#         self.snapshot_path = snapshot_path
#         self.model = DDP(self.model, device_ids=[self.gpu_id])

#     def train(self, max_epochs):
#         for epoch in range(max_epochs):
#             self.model.train()
#             self.train_data.sampler.set_epoch(epoch)
#             for source, targets in self.train_data:
#                 source = source.to(self.gpu_id)
#                 targets = targets.to(self.gpu_id)
#                 self.optimizer.zero_grad()
#                 output = self.model(source)
#                 loss = nn.functional.cross_entropy(output, targets)
#                 loss.backward()
#                 self.optimizer.step()

# def load_train_objs():
#     # Initialize dataset and model
#     data_path = config.multi_speaker_data_path
#     with open(f'{data_path}/index.pkl', 'rb') as f:
#         index = pickle.load(f)
#     dataset = MultispeakerDataset(index, data_path)
#     model = vqvae.Model(rnn_dims=896, fc_dims=896, global_decoder_cond_dims=dataset.num_speakers(), 
#                         upsample_factors=(4, 4, 4), normalize_vq=True, noise_x=True, noise_y=True).cuda()
#     optimizer = optim.AdamW(model.parameters(), lr=1e-4)
#     return dataset, model, optimizer

# def prepare_dataloader(dataset, batch_size):
#     # Create DataLoader
#     return DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False,
#                       sampler=DistributedSampler(dataset))

# def main(save_every, total_epochs, batch_size, snapshot_path="snapshot.pt"):
#     ddp_setup()
#     dataset, model, optimizer = load_train_objs()
#     train_data = prepare_dataloader(dataset, batch_size)
#     trainer = Trainer(model, train_data, optimizer, save_every, snapshot_path)
#     trainer.train(total_epochs)
#     destroy_process_group()

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='Distributed training for VQVAE model')
#     parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
#     parser.add_argument('save_every', type=int, help='How often to save a snapshot')
#     parser.add_argument('--batch_size', default=256, type=int, help='Input batch size on each device')
#     args = parser.parse_args()
    
#     main(args.save_every, args.total_epochs, args.batch_size)

