import numpy as np
import random
import argparse
import datetime
import dateutil.tz
import os

from miscc.config import cfg, cfg_from_file
from nltk.tokenize import RegexpTokenizer
from miscc.utils import mkdir_p
from collections import defaultdict

from DAMSM import RNN_ENCODER
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.transforms as transforms
from model import NetG,NetD

def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/coco_streamlit.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    
    args = parser.parse_args()
    return args

def setup():
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id
    cfg.DATA_DIR = args.data_dir
    args.manualSeed = 100

    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)

    #blah

    text_encoder = RNN_ENCODER(27297, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    for p in text_encoder.parameters():
        p.requires_grad = False
    text_encoder.eval()    

    return text_encoder, netG

    

def single_sampling(text_encoder, netG, caption, device):
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    netG.load_state_dict(torch.load('models/%s/netG.pth'%(cfg.CONFIG_NAME), map_location='cuda:0'))
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE #1
    s_tmp = model_dir
    save_dir = '%s/%s' % (s_tmp, split_dir)
    mkdir_p(save_dir)

    hidden = text_encoder.init_hidden(batch_size)
    caption, cap_lens = prepare_data(caption)
    words_embs, sent_emb = text_encoder(caption, cap_lens, hidden)
    words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

    with torch.no_grad():
        noise = torch.randn(batch_size, 100)
        noise=noise.to(device)
        fake_imgs = netG(noise,sent_emb)

    s_tmp = '%s/single' % (save_dir)
    folder = s_tmp[:s_tmp.rfind('/')]
    if not os.path.isdir(folder):
        print('Make a new folder: ', folder)
        mkdir_p(folder)
    im = fake_imgs.data.cpu().numpy()

    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    #im = np.transpose(im, (1, 2, 0))
    im = np.transpose(im, axes=[2, 3, 0, 1])
    print(np.shape(im))
    im = im.squeeze()
    print(np.shape(im))
    im = Image.fromarray(im)
    fullpath = '%s.png' % (s_tmp)
    im.save(fullpath)
    return im

def prepare_data(caption):
    caption_lens = len(caption)
    sorted_cap_lens = [caption_lens]
    sorted_cap_indices = 0
    # sorted_cap_lens, sorted_cap_indices = \
    # torch.sort(torch.from_numpy(np.array(caption_lens)), 0, True)

    # print(caption)
    # print(sorted_cap_lens)
    # print(sorted_cap_lens)
    # caption = caption[sorted_cap_indices].squeeze()
    if cfg.CUDA:
        #caption = Variable(torch.FloatTensor(caption)).cuda()
        #caption = Variable(torch.from_numpy(np.array(caption))).cuda()
        caption = torch.from_numpy(np.array([caption])).cuda()
        #sorted_cap_lens = Variable(sorted_cap_lens).cuda()
    else:
        #caption = Variable(torch.FloatTensor(caaption))
        #caption = Variable(torch.from_numpy(np.array(caption))).cuda()
        caption = torch.from_numpy(np.array(caption))
        #sorted_cap_lens = Variable(sorted_cap_lens)
    
    print([caption.to(torch.int64), sorted_cap_lens])
    print(type(caption))
    return [caption.to(torch.int64), sorted_cap_lens]

def get_data(caption):
    # self.transform = transform
    # self.norm = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # self.target_transform = target_transform
    # self.embeddings_num = cfg.TEXT.CAPTIONS_PER_IMAGE

    # self.imsize = []
    #     for i in range(cfg.TREE.BRANCH_NUM):
    #         self.imsize.append(base_size)
    #         base_size = base_size * 2

    # self.data = []
    # self.bbox = None

    #blah
    cap = caption.replace("\ufffd\ufffd", " ")
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(cap.lower())

    if len(tokens) == 0:
        print('cap', cap)

    tokens_new = []
    for t in tokens:
        t = t.encode('ascii', 'ignore').decode('ascii')
        if len(t) > 0:
            tokens_new.append(t)
    
    return tokens_new

def build_dictionary(caption):
    print("CAPTION")
    print(caption)
    word_counts = defaultdict(float)
    for word in caption:
        word_counts[word] += 1

    vocab = [w for w in word_counts if word_counts[w] >= 0]
    ixtoword = {}
    ixtoword[0] = '<end>'
    wordtoix = {}
    wordtoix['<end>'] = 0
    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    rev = []
    for w in caption:
        if w in wordtoix:
            rev.append(wordtoix[w])
    print("REV")
    print(rev)
    return rev


# def single_inference(description, netG):
#     st_motion_input = description
#     st_content_input_shape = [1, 1, description.shape[-1]]
#     st_content_input = torch.zeros(st_content_input_shape)
#     st_motion_input = st_motion_input.unsqueeze(0)

#     if cfg.CUDA:
#         st_motion_input = st_motion_input.cuda()
#         st_content_input = st_content_input.cuda()
    
#     lr_fake, st_fake = self.sample_stories(st_motion_input, st_content_input)

# def sample_stories(st_motion_input, st_content_input, netG):
#     lr_fake, st_fake, _, _, _, _ = netG.sample_videos(st_motion_input, st_content_input)
#     return lr_fake, st_fake

# def sample_videos()