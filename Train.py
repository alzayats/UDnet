"""Train UDnet on a folder of underwater images (no ground truth needed).

    python Train.py --train_dir path/to/images
    python Train.py --train_dir path/to/images --train_list splits/my_train.txt
"""
import os
import sys
import torch
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lrs
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_utils.UDnet import mynet
from model_utils.data import get_training_set


def str2bool(v):
    # argparse type=bool is broken: bool("False") is True. Parse the string properly.
    return str(v).lower() in ("true", "1", "yes", "t", "y")


# Training settings
parser = argparse.ArgumentParser(description='Train UDnet')
parser.add_argument('--train_dir', type=str, default=None,
                    help='folder of training images (default: <data_dir>/<data_train_dataset>)')
parser.add_argument('--train_list', type=str, default=None,
                    help='optional text file with one image file name per line; only these are used')
parser.add_argument('--data_dir', type=str, default='./dataset/')
parser.add_argument('--data_train_dataset', type=str, default='image')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batchSize', type=int, default=6, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=100, help='save epoch_<N>.pth every N epochs')
parser.add_argument('--start_iter', type=int, default=1, help='starting epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
parser.add_argument('--patch_size', type=int, default=256, help='Size of cropped image')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--gpu_mode', type=str2bool, default=True, help='set False to run on CPU')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--decay', type=int, default=10000,
                    help='multiply the learning rate by --gamma every N epochs (default: no decay)')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--data_augmentation', type=str2bool, default=True)
parser.add_argument('--resume_train', type=str2bool, default=False,
                    help='continue from <save_folder>/resume.pth (model, optimizer and epoch)')
parser.add_argument('--pretrained', default=None,
                    help='initialise from a model, e.g. weights/UDnet.pth, to fine-tune on your data')


def train(epoch):
    epoch_loss = 0
    model.train()
    bar = tqdm(training_data_loader, desc=f'Epoch {epoch}/{opt.nEpochs}', leave=False)
    for input, target, _ in bar:
        input = input.to(device)
        target = target.to(device)

        model.forward(input, target, training=True)
        loss = model.elbo(target)
        optimizer.zero_grad()
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()
        bar.set_postfix(loss=f'{loss.item():.4f}', lr=optimizer.param_groups[0]['lr'])
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))


def checkpoint(epoch):
    model_out_path = os.path.join(opt.save_folder, "epoch_{}.pth".format(epoch))
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def save_resume(epoch):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(), 'epoch': epoch},
               os.path.join(opt.save_folder, 'resume.pth'))


def load_weights(path):
    state = torch.load(path, map_location='cpu')
    if isinstance(state.get('model'), dict):
        state = state['model']
    state = {k: v for k, v in state.items() if not k.startswith('decoder.physnet.')}
    model.load_state_dict(state, strict=True)


if __name__ == '__main__':
    opt = parser.parse_args()
    if not opt.gpu_mode:
        opt.device = 'cpu'          # --gpu_mode False -> CPU (mynet builds on opt.device)
    elif not torch.cuda.is_available():
        sys.exit("No GPU found. Run with --gpu_mode False to train on the CPU.")
    device = torch.device(opt.device)
    cudnn.benchmark = True
    print(opt)

    torch.manual_seed(opt.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(opt.seed)

    train_dir = opt.train_dir or os.path.join(opt.data_dir, opt.data_train_dataset)
    if not os.path.isdir(train_dir):
        sys.exit(f"Training folder not found: {train_dir}. Point --train_dir at a folder of images.")
    print('===> Loading datasets from', train_dir)
    train_set = get_training_set(train_dir, opt.patch_size, opt.data_augmentation, opt.train_list)
    print(f'===> {len(train_set)} training images')
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                      shuffle=True, drop_last=len(train_set) > opt.batchSize)

    model = mynet(opt)
    if opt.pretrained:
        load_weights(opt.pretrained)
        print('Initialised from', opt.pretrained)

    optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    milestones = [i for i in range(1, opt.nEpochs + 1) if i % opt.decay == 0]
    scheduler = lrs.MultiStepLR(optimizer, milestones, opt.gamma)

    os.makedirs(opt.save_folder, exist_ok=True)
    if opt.resume_train:
        resume = torch.load(os.path.join(opt.save_folder, 'resume.pth'), map_location='cpu')
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        opt.start_iter = resume['epoch'] + 1
        print('Resuming from epoch', opt.start_iter)

    for epoch in range(opt.start_iter, opt.nEpochs + 1):
        train(epoch)
        scheduler.step()

        if epoch % opt.snapshots == 0:
            checkpoint(epoch)
        if epoch % 2 == 0 or epoch == opt.nEpochs:
            checkpoint("last")
            save_resume(epoch)
