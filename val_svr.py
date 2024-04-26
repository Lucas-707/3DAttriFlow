import argparse
import torch
import munch
import yaml
# from dataset_svr.trainer_dataset import build_dataset_val
from dataset_svr.dataset_clevr import build_dataset_val

import torch
from utils.train_utils import *
import logging
import importlib
import random
import munch
import yaml
import os
import argparse
from utils.model_utils import *
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def val():
    
    dataloader_test = build_dataset_val(args)

    if not args.manual_seed:
        seed = random.randint(1, 10000)
    else:
        seed = int(args.manual_seed)
    logging.info('Random Seed: %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_module = importlib.import_module('.%s' % args.model_name, 'models')
    net = torch.nn.DataParallel(model_module.Model(args))
    net.cuda()
    if hasattr(model_module, 'weights_init'):
        net.module.apply(model_module.weights_init)
    
    ckpt = torch.load(args.load_model)
    net.module.load_state_dict(ckpt['net_state_dict'])
    logging.info("%s's previous weights loaded." % args.model_name)

    net.module.eval()

    logging.info('Testing...')

    test_loss_l1 = AverageValueMeter()
    test_loss_l2 = AverageValueMeter()

    with tqdm(dataloader_test) as t:
        for i, data in enumerate(t):
            # data: ['pointcloud_path', 'image_path', 'name', 'category', 'points', 'image']
            with torch.no_grad():
                # print("data batch keys: ", data.keys())
                # print("point path: ", data['pointcloud_path'])
                images = data['image'].cuda() # Bx3x224x224
                gt = data['points'].cuda()

                batch_size = gt.shape[0]
                
                pred_points = net(images)
                # print("pred_points: ", pred_points.shape)
                pc_path = data['pointcloud_path'][0]
                pc_ids = pc_path.split('/')[-3:]
                output_dir = "/home/yuwu3/SpaceQA/Code/Preprocess/3DAttriFlow/output"
                save_dir = os.path.join( *([output_dir]+pc_ids[:-1]) )
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_path = os.path.join( *([output_dir]+pc_ids) )
                # print("save path: ", save_path)
                torch.save(pred_points, save_path)


                loss_p, loss_t = calc_cd(pred_points, gt)

                cd_l1_item = torch.sum(loss_p).item() / batch_size
                cd_l2_item = torch.sum(loss_t).item() / batch_size
                test_loss_l1.update(cd_l1_item, images.shape[0])
                test_loss_l2.update(cd_l2_item, images.shape[0])

    print('cd_l1 %f cd_l2 %f' % (test_loss_l1.avg, test_loss_l2.avg))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    parser.add_argument('-gpu', '--gpu_id', help='gpu_id', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(arg.gpu_id)
    print('Using gpu:' + str(arg.gpu_id))

    val()
