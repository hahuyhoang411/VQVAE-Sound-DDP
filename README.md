# WaveRNN + VQ-VAE

This is a Pytorch implementation of [WaveRNN](
https://arxiv.org/abs/1802.08435v1). Currently 3 top-level networks are
provided:

* A [VQ-VAE](https://avdnoord.github.io/homepage/vqvae/) implementation with a
  WaveRNN decoder. Trained on a multispeaker dataset of speech, it can
  demonstrate speech reconstruction and speaker conversion.
* A vocoder implementation. Trained on a single-speaker dataset, it can turn a
  mel spectrogram into raw waveform.
* An unconditioned WaveRNN. Trained on a single-speaker dataset, it can generate
  random speech.

[Audio samples](https://mkotha.github.io/WaveRNN/).

It has been tested with the following datasets.

Multispeaker datasets:

* [VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651)

Single-speaker datasets:

* [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)

## Preparation

### Requirements

* Python 3.6 or newer
* PyTorch with CUDA enabled
* [librosa](https://github.com/librosa/librosa)
* [apex](https://github.com/NVIDIA/apex) if you want to use FP16 (it probably
  doesn't work that well).


### Create config.py

```
cp config.py.example config.py
```

### Preparing VCTK

You can skip this section if you don't need a multi-speaker dataset.

1. Download and uncompress [the VCTK dataset](
  https://datashare.is.ed.ac.uk/handle/10283/2651).
2. `python preprocess_multispeaker.py /path/to/dataset/VCTK-Corpus/wav48
  /path/to/output/directory`
3. In `config.py`, set `multi_speaker_data_path` to point to the output
  directory.

### Preparing LJ-Speech

You can skip this section if you don't need a single-speaker dataset.

1. Download and uncompress [the LJ speech dataset](
  https://keithito.com/LJ-Speech-Dataset/).
2. `python preprocess16.py /path/to/dataset/LJSpeech-1.1/wavs
  /path/to/output/directory`
3. In `config.py`, set `single_speaker_data_path` to point to the output
  directory.

## Usage

`wavernn.py` is the entry point:

```
$ python wavernn.py
```

By default, it trains a VQ-VAE model. The `-m` option can be used to tell the
the script to train a different model.

Trained models are saved under the `model_checkpoints` directory.

By default, the script will take the latest snapshot and continues training
from there. To train a new model freshly, use the `--scratch` option.

Every 50k steps, the model is run to generate test audio outputs. The output
goes under the `model_outputs` directory.

When the `-g` option is given, the script produces the output using the saved
model, rather than training it.

# Deviations from the papers

I deviated from the papers in some details, sometimes because I was lazy, and
sometimes because I was unable to get good results without it. Below is a
(probably incomplete) list of deviations.

All models:

* The sampling rate is 22.05kHz.

VQ-VAE:

* I normalize each latent embedding vector, so that it's on the unit 128-
  dimensional sphere. Without this change, I was unable to get good utilization
  of the embedding vectors.
* In the early stage of training, I scale with a small number the penalty term
  that apply to the input to the VQ layer. Without this, the input very often
  collapses into a degenerate distribution which always selects the same
  embedding vector.
* During training, the target audio signal (which is also the input signal) is
  translated along the time axis by a random amount, uniformly chosen from
  [-128, 127] samples. Less importantly, some additive and multiplicative
  Gaussian noise is also applied to each audio sample. Without these types of
  noise, the feature captured by the model tended to be very sensitive to small
  purterbations to the input, and the subjective quality of the model output
  kept descreasing after a certain point in training.
* The decoder is based on WaveRNN instead of WaveNet. See the next section for
  details about this network.

# Context stacks

The VQ-VAE implementation uses a WaveRNN-based decoder instead of a WaveNet-
based decoder found in the paper. This is a WaveRNN network augmented
with a context stack to extend the receptive field.  This network is
defined in `layers/overtone.py`.

The network has 6 convolutions with stride 2 to generate 64x downsampled
'summary' of the waveform, and then 4 layers of upsampling RNNs, the last of
which is the WaveRNN layer. It also has U-net-like skip connections that
connect layers with the same operating frequency.

# Acknowledgement

The code is based on [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN).

# Code change

## Write output
Old code:
```
for i, x in enumerate(gt) :
    librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x.cpu().numpy(), sr=sample_rate)
    audio = out[i][:len(x)].cpu().numpy()
    librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, sr=sample_rate)
    audio_tr = out[n_points+i][:len(x)].cpu().numpy()
    librosa.output.write_wav(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, sr=sample_rate)
```

New code:
```
# librosa is outdated
for i, x in enumerate(gt) :
    sf.write(f'{paths.gen_path()}/{k}k_steps_{i}_target.wav', x.cpu().numpy(), samplerate=sample_rate)
    audio = out[i][:len(x)].cpu().numpy()
    sf.write(f'{paths.gen_path()}/{k}k_steps_{i}_generated.wav', audio, samplerate=sample_rate)
    audio_tr = out[n_points+i][:len(x)].cpu().numpy()
    sf.write(f'{paths.gen_path()}/{k}k_steps_{i}_transferred.wav', audio_tr, samplerate=sample_rate)
```

## DDP changes

### DDP Wraper

Init Distributed GPU
```
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# Wrap all of the training code into main()
def main(rank, world_size):
    setup(rank, world_size)
    # Training code
    # ...
    # ...

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
```

### Data Loader
Old code:
```
for e in range(epochs) :

    trn_loader = DataLoader(dataset, collate_fn=lambda batch: env.collate_multispeaker_samples(pad_left, window, pad_right, batch), batch_size=batch_size,
                            num_workers=2, shuffle=True, pin_memory=True)

    start = time.time()
    running_loss_c = 0.
    running_loss_f = 0.
    running_loss_vq = 0.
    running_loss_vqc = 0.
    running_entropy = 0.
    running_max_grad = 0.
    running_max_grad_name = ""

    iters = len(trn_loader)
```

New code:
```
# Convert to partial so can split to multiple GPU
for e in range(epochs) :
    collate_fn_with_padding = partial(env.collate_multispeaker_samples, pad_left, window, pad_right)
    trn_loader = DataLoader(dataset, collate_fn=collate_fn_with_padding, batch_size=batch_size,
                            num_workers=8, shuffle=False, pin_memory=True, sampler=DistributedSampler(dataset))

    start = time.time()
    running_loss_c = 0.
    running_loss_f = 0.
    running_loss_vq = 0.
    running_loss_vqc = 0.
    running_entropy = 0.
    running_max_grad = 0.
    running_max_grad_name = ""

    iters = len(trn_loader)
```

Old code:
```
def restore(path, model):
    model.load_state_dict(torch.load(path))

    match = re.search(r'_([0-9]+)\.pyt', path)
    if match:
        return int(match.group(1))

    step_path = re.sub(r'\.pyt', '_step.npy', path)
    return np.load(step_path)
```

New code:

```
# Handle the case where the model is wrapped with DDP
def restore(path, model):
    state_dict = torch.load(path)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = {f'module.{k}': v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    match = re.search(r'_([0-9]+)\.pyt', path)
    if match:
        return int(match.group(1))

    step_path = re.sub(r'\.pyt', '_step.npy', path)
    return np.load(step_path)
```

Old code:
```
class MultispeakerDataset(Dataset):
    def __init__(self, index, path):
        self.path = path
        self.index = index
        self.all_files = [(i, name) for (i, speaker) in enumerate(index) for name in speaker]

    def __getitem__(self, index):
        speaker_id, name = self.all_files[index]
        speaker_onehot = (np.arange(len(self.index)) == speaker_id).astype(np.long)
        audio = np.load(f'{self.path}/{speaker_id}/{name}.npy')
        return speaker_onehot, audio

    def __len__(self):
        return len(self.all_files)

    def num_speakers(self):
        return len(self.index)
```

New code:
```
# Change np.long to np._int since long is outdated 
class MultispeakerDataset(Dataset):
    def __init__(self, index, path):
        self.path = path
        self.index = index
        self.all_files = [(i, name) for (i, speaker) in enumerate(index) for name in speaker]

    def __getitem__(self, index):
        speaker_id, name = self.all_files[index]
        speaker_onehot = (np.arange(len(self.index)) == speaker_id).astype(np.int_)
        audio = np.load(f'{self.path}/{speaker_id}/{name}.npy')
        return speaker_onehot, audio

    def __len__(self):
        return len(self.all_files)

    def num_speakers(self):
        return len(self.index)
```

## Bf16 changes

Old code:
```
def forward_generate(self, global_decoder_cond, samples, deterministic=False, use_half=False, verbose=False):
    if use_half:
        samples = samples.half()
    # samples: (L)
    #logger.log(f'samples: {samples.size()}')
    self.eval()
    with torch.no_grad() :
        continuous = self.encoder(samples)
        discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
        logger.log(f'entropy: {entropy}')
        # cond: (1, L1, 64)
        #logger.log(f'cond: {cond.size()}')
        output = self.overtone.generate(discrete.squeeze(2), global_decoder_cond, use_half=use_half, verbose=verbose)
    self.train()
    return output
```

New code:
```
def forward_generate(self, global_decoder_cond, samples, deterministic=False, use_half=False, verbose=False):
    if use_half:
        samples = samples.to(dtype=torch.bfloat16) # New add for bf16
    # samples: (L)
    #logger.log(f'samples: {samples.size()}')
    self.eval()
    with torch.no_grad() :
        samples = samples.to(dtype=torch.bfloat16) # New add for bf16
        continuous = self.encoder(samples)
        discrete, vq_pen, encoder_pen, entropy = self.vq(continuous.unsqueeze(2))
        logger.log(f'entropy: {entropy}')
        # cond: (1, L1, 64)
        #logger.log(f'cond: {cond.size()}')
        output = self.overtone.generate(discrete.squeeze(2), global_decoder_cond, use_half=use_half, verbose=verbose)
    self.train()
    return output
```

Old code:
```
if use_half:
    import apex
    optimiser = apex.fp16_utils.FP16_Optimizer(optimiser, dynamic_loss_scale=True)
```

New code:
```
if use_half:
    import apex
    optimiser = optimiser # Use direct optimizer
```

<table>
<tr>
<th>Old Code</th>
<th>New Code</th>
</tr>
<tr>
<td>

```python
p_cf, vq_pen, encoder_pen, entropy = self(speaker, x, translated)
p_c, p_f = p_cf
loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse)
loss_f = criterion(p_f.transpose(1, 2).float(), y_fine)
```
```python
# Long format requirement
p_cf, vq_pen, encoder_pen, entropy = self(speaker, x, translated)
p_c, p_f = p_cf
loss_c = criterion(p_c.transpose(1, 2).float(), y_coarse.long())
loss_f = criterion(p_f.transpose(1, 2).float(), y_fine.long())
```
