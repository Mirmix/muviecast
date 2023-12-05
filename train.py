import os
from opt import get_opts
from utils import *
from datasets.custom import CustomDataset
from datasets.patchmatch_dataloader import MVSDataset
import torch
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True  # this increases inference speed a little
from models.style import *
from models.adain import AdainNST, compute_mean_std
from models.patchmatchnet import PatchmatchNet
from models.mvsnet import CascadeMVSNet
from utils import load_ckpt
from datasets.utils import print_args
from inplace_abn import ABN
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import kornia as K  # for image structure loss

flag_image_structure_loss = True


class StyleSystem(LightningModule):
    def __init__(self, hparams):
        """ Initializes the model """
        super(StyleSystem, self).__init__()
        self.hp = hparams
        self.root_dir = self.hp.root_dir
        self.train_dataset = None
        self.val_dataset = None

        # Hyperparameters for training
        self.lambda_style = self.hp.lambda_style
        self.lambda_content = self.hp.lambda_content
        self.lambda_volume = self.hp.lambda_volume
        self.lambda_depth = self.hp.lambda_depth
        self.lambda_structure = self.hp.lambda_structure
        self.lambda_nnfm = self.hp.lambda_nnfm
        self.sample_interval = self.hp.sample_interval
        self.epochs = self.hp.num_epochs
        self.ablation_suffix = self.hp.ablation_suffix
        self.image_size = self.hp.img_wh
        self.image_extension = self.hp.img_ext
        self.lr = hparams.lr
        self.n_views = hparams.n_views
        self.l2_loss = torch.nn.MSELoss()
        self.smooth_l1_loss = torch.nn.SmoothL1Loss()

        self.ckpt_dir = self.hp.ckpt_dir
        self.patchmatch_weights = self.hp.patchmatch_weights
        self.adain_weights = self.hp.adain_weights
        self.unet_weights = self.hp.unet_weights
        self.casmvs_weights = self.hp.casmvs_weights

        # useful cache for style image Nx3xHxW
        self.style_dir = self.hp.style_dir
        self.style_name = self.hp.style_name
        self.scan_name = self.hp.scan_name
        self.output_dir = self.hp.output_dir

        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.style = self.image_transform(self.image_size)(
            Image.open("%s/%s.jpg" % (self.style_dir, self.style_name))).repeat(
            self.n_views, 1, 1, 1)

        self.unet_weights = self.hp.unet_weights
        # transformations for normalizing and unnormalizing images, Tensor -> PIL -> Tensor
        self.normalizeImg = T.Compose(
            [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.normalizeTensor = T.Compose([T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.unnormalizeTensor2Img = T.Compose([T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]), T.ToPILImage()])
        self.unnormalizeTensor = T.Compose([T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                                        std=[1 / 0.229, 1 / 0.224, 1 / 0.225])])
        self.unnormalizeTensor2PIL2Tensor = T.Compose(
            [T.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]),
             T.ToPILImage(), T.ToTensor()])

        # models
        # Backbone options
        self.use_adain = self.hp.use_adain  # use AdaIN with VGG19
        self.use_patchmatchnet = not self.hp.use_casmvsnet  # use PatchmatchNet

        # Setting up the MVS backbone
        self.mvsnet = None

        if self.use_patchmatchnet:

            self.mvsnet = PatchmatchNet(
                patchmatch_interval_scale=self.hp.patchmatch_interval_scale,
                propagation_range=self.hp.patchmatch_range,
                patchmatch_iteration=self.hp.patchmatch_iteration,
                patchmatch_num_sample=self.hp.patchmatch_num_sample,
                propagate_neighbors=self.hp.propagate_neighbors,
                evaluate_neighbors=self.hp.evaluate_neighbors
            )
            state_dict = torch.load(self.ckpt_dir + self.patchmatch_weights)["model"]
            self.mvsnet.load_state_dict(state_dict, strict=False)

        else:
            if self.hp.num_groups > 1:  # use group normalization for memory efficiency
                self.mvsnet = CascadeMVSNet(n_depths=[8, 32, 48],
                                            interval_ratios=[1.0, 2.0, 4.0],
                                            num_groups=self.hp.num_groups,
                                            norm_act=ABN)
                load_ckpt(self.mvsnet, self.ckpt_dir + self.casmvs_weights)
            else:
                self.mvsnet = CascadeMVSNet(n_depths=[8, 32, 48],
                                            interval_ratios=[1.0, 2.0, 4.0],
                                            norm_act=ABN)
                load_ckpt(self.mvsnet, self.ckpt_dir + self.casmvs_weights)

        self.levels = self.hp.levels
        self.mvsnet.requires_grad_(False)

        # Setting up the style transfer backbone
        self.transformer = None
        self.decoder = None
        self.vgg = None
        self.adain_net = None
        if self.use_adain:
            self.adain_net = AdainNST(self.ckpt_dir + self.adain_weights)
            self.vgg = self.adain_net.encoder
            self.decoder = self.adain_net.decoder
        else:
            self.transformer = TransformerNet()
            load_ckpt(self.transformer,
                      '%s/%s' % (self.ckpt_dir, self.unet_weights)) if self.unet_weights != "None" else None
            self.vgg = VGG16(requires_grad=False)
            self.features_style = self.vgg(self.style)
            self.gram_style = [self.gram_matrix(y) for y in self.features_style]
        # if num gpu is 1, print model structure and number of params
        print("Num of gpus: ", self.hp.num_gpus)
        if self.hp.num_gpus == 1 and self.transformer is not None:
            print('number of parameters : %.2f M' %
                  (sum(p.numel() for p in self.transformer.parameters() if p.requires_grad) / 1e6))

        # color transfer
        self.pre_color_transfer = self.hp.color_adjust
        self.post_color_transfer = self.hp.color_adjust
        self.ct_mat = None
        self.color_steps = 1
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def gram_matrix(self, y):
        """ Returns the gram matrix of y (used to compute style loss) """
        (b, c, h, w) = y.size()
        features = y.view(b, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def argmin_cos_distance(self, a, b, center=False):
        """
        a: [b, c, hw],
        b: [b, c, h2w2]
        """
        if center:
            a = a - a.mean(2, keepdims=True)
            b = b - b.mean(2, keepdims=True)

        b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
        b = b / (b_norm + 1e-8)

        z_best = []
        loop_batch_size = int(1e8 / b.shape[-1])
        for i in range(0, a.shape[-1], loop_batch_size):
            a_batch = a[..., i: i + loop_batch_size]
            a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
            a_batch = a_batch / (a_batch_norm + 1e-8)

            d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)

            z_best_batch = torch.argmin(d_mat, 2)
            z_best.append(z_best_batch)
        z_best = torch.cat(z_best, dim=-1)

        return z_best

    def nn_feat_replace(self, a, b):
        n, c, h, w = a.size()
        n2, c2, h2, w2 = b.size()
        a_flat = a.view(n, c, -1)
        b_flat = b.view(n2, c, -1)
        b_ref = b_flat.clone()

        z_new = []
        for i in range(n):
            z_best = self.argmin_cos_distance(a_flat[i: i + 1], b_flat[i: i + 1])
            z_best = z_best.unsqueeze(1).repeat(1, c, 1)
            feat = torch.gather(b_ref, 2, z_best)
            z_new.append(feat)

        z_new = torch.cat(z_new, 0)
        z_new = z_new.view(n, c, h, w)
        return z_new

    def cos_loss(self, a, b):
        a_norm = (a * a).sum(1, keepdims=True).sqrt()
        b_norm = (b * b).sum(1, keepdims=True).sqrt()
        a_tmp = a / (a_norm + 1e-8)
        b_tmp = b / (b_norm + 1e-8)
        cossim = (a_tmp * b_tmp).sum(1)
        cos_d = 1.0 - cossim
        return cos_d.mean()

    def decode_batch(self, batch):
        """ Decodes batch into image, camera and depth tensors """
        if self.use_patchmatchnet:
            # Decode batch for PatchmatchNet
            imgs = batch['images']
            intrinsics = batch['intrinsics']
            extrinsics = batch['extrinsics']
            depth_min = batch['depth_min'].item()
            depth_max = batch['depth_max'].item()
            if self.pre_color_transfer:
                imgs = torch.stack(imgs, dim=1).squeeze(0).permute(0, 2, 3, 1)  # [V, H, W, C]
                imgs = apply_color_tf(imgs, self.ct_mat).permute(0, 3, 1, 2)  # [V, C, H, W]
                imgs = torch.split(imgs, 1, dim=0)

            return imgs, intrinsics, extrinsics, depth_min, depth_max
        else:
            # Decode batch for CasMVSNet

            # imgs: (B, V, 3, H, W)
            # proj_mats: (B, V-1, self.levels, 3, 4) from fine to coarse
            # init_depth_min, depth_interval: (B) or float

            imgs = batch['images']
            proj_mats = batch['proj_mats']
            init_depth_min = batch['init_depth_min'].item()
            depth_interval = batch['depth_interval'].item()
            if self.pre_color_transfer:
                imgs = apply_color_tf(self.unnormalizeTensor(imgs).squeeze(0).permute(0, 2, 3, 1), self.ct_mat)
                imgs = self.normalizeTensor(imgs.permute(0, 3, 1, 2)).unsqueeze(0)

            return imgs, proj_mats, init_depth_min, depth_interval

    def image_transform(self, image_size=None):
        """ Transforms for image to zero mean and unit variance """

        width, height = image_size
        resize = [transforms.Resize((height, width))] if image_size else []
        transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        return transform

    def train_dataloader(self):
        """ Returns training dataloader """
        if self.use_patchmatchnet:
           self.train_dataset = MVSDataset(data_path=self.root_dir,
                                                num_views=self.n_views - 1,
                                                scan_name=self.scan_name,
                                                num_light_idx=-1,
                                                image_extension=self.image_extension,
                                                img_wh=self.image_size
                                                )
        else:
            self.train_dataset = CustomDataset(self.root_dir, '', True, n_views=self.n_views,
                                                   img_wh=self.image_size, scan_name=self.scan_name,extension=self.image_extension)
        if self.pre_color_transfer:
            self.ct_mat = self.get_color_transfer_matrix()
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=2,
                          persistent_workers=True,
                          batch_size=self.hp.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        """ Returns validation dataloader """
        if self.use_patchmatchnet:
            self.val_dataset = MVSDataset(data_path=self.root_dir,
                                              num_views=self.n_views - 1,
                                              scan_name=self.scan_name,
                                              num_light_idx=-1,
                                              image_extension=self.image_extension,
                                              img_wh=self.image_size)
        else:
            self.val_dataset = CustomDataset(self.root_dir, '', False, n_views=self.n_views,
                                                 img_wh=self.image_size, scan_name=self.scan_name, extension = self.image_extension)
        # take a sample from the dataset
        self.val_dataset = [self.val_dataset[i] for i in range(1)]
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=2,
                          persistent_workers=True,
                          batch_size=self.hp.batch_size,
                          pin_memory=True)

    def forward(self, imgs, proj_mats=None, init_depth_min=None, depth_interval=None,
                intrinsics=None, extrinsics=None, depth_min=None, depth_max=None):
        """ Forward pass of the model """
        # squeeze the batch dimension
        content_loss = torch.tensor(0.0).cuda()
        style_loss = torch.tensor(0.0).cuda()
        volume_loss = torch.tensor(0.0).cuda()
        depth_loss = torch.tensor(0.0).cuda()
        structure_loss = torch.tensor(0.0).cuda()
        nnfm_loss = torch.tensor(0.0).cuda()

        if self.use_patchmatchnet:
            # Dataloader of PatchmatchNet returns a list of tensors
            # PatchmatchNet trained on 0-1 range images
            imgs = torch.stack(imgs, dim=1).squeeze(0)  # list to tensor conversion V, 3, H, W
            for i in range(imgs.shape[0]):
                imgs[i] = self.normalizeImg(transforms.ToPILImage()(imgs[i]))  # normalizing the image
            imgs = imgs.unsqueeze(0)  # add batch dimension

        images_transformed = None
        # get the features of the both content and style image
        if self.use_adain:
            # print("Running AdaIN for style transfer")

            images_transformed, t = self.adain_net(imgs.squeeze(0), self.style.cuda(), 1., True)
            stylized_content_feats = self.adain_net.encoder_forward(
                images_transformed, True
            )
            stylized_feats = self.adain_net.encoder_forward(images_transformed)
            style_feats = self.adain_net.encoder_forward(self.style.cuda())
            content_loss = self.lambda_content * F.mse_loss(t, stylized_content_feats)

            style_loss = 0
            for stz, sty in zip(stylized_feats, style_feats):
                stz_m, stz_s = compute_mean_std(stz)
                sty_m, sty_s = compute_mean_std(sty)
                style_loss += F.mse_loss(stz_m, sty_m) + F.mse_loss(stz_s, sty_s)
            style_loss = self.lambda_style * style_loss

        else:

            # print("Running UNet for style transfer")

            # imgs: (B, V, 3, H, W)
            # proj_mats: (B, V-1, self.levels, 3, 4) from fine to coarse
            # init_depth_min, depth_interval: (B) or float
            images_transformed = self.transformer(imgs.squeeze(0))
            features_original = self.vgg(imgs.squeeze(0))
            features_transformed = self.vgg(images_transformed)

            # content loss (L2 loss between original and transformed features)
            if self.lambda_content > 0:
                content_loss = self.lambda_content * self.smooth_l1_loss(features_transformed.relu2_2,
                                                                         features_original.relu2_2)
            # Compute style loss as MSE between gram matrices
            if self.lambda_style > 0:
                for ft_y, gm_s in zip(features_transformed, self.gram_style):
                    gm_y = self.gram_matrix(ft_y)
                    style_loss += self.l2_loss(gm_y, gm_s.cuda())
                style_loss *= self.lambda_style
                del gm_y, gm_s, ft_y

            if self.lambda_nnfm > 0:
                for i in range(len(features_transformed)):
                    target_feats = self.nn_feat_replace(features_transformed[i], self.features_style[i].cuda())
                    nnfm_loss += self.cos_loss(features_transformed[i], target_feats) * self.lambda_nnfm
                    del target_feats

            # Clean up
            del features_original, features_transformed,

        if flag_image_structure_loss:
            # We use the grayscale image as the input of the structure loss
            # CHECK THE GRAYSCALE IMAGES FOR BOTH
            sobel_img = K.filters.sobel(K.color.rgb_to_grayscale(self.unnormalizeTensor(imgs.squeeze(0))))
            structure_loss += self.smooth_l1_loss(sobel_img, K.filters.sobel(
                K.color.rgb_to_grayscale(self.unnormalizeTensor(images_transformed))))
            laplacian_img = K.filters.laplacian(K.color.rgb_to_grayscale(self.unnormalizeTensor(imgs.squeeze(0))),
                                                kernel_size=5)
            structure_loss += self.smooth_l1_loss(laplacian_img, K.filters.laplacian(
                K.color.rgb_to_grayscale(self.unnormalizeTensor(images_transformed)), kernel_size=5))
            canny_img = K.filters.canny(K.color.rgb_to_grayscale(self.unnormalizeTensor(imgs.squeeze(0))))[0]
            structure_loss += self.smooth_l1_loss(canny_img, K.filters.canny(
                K.color.rgb_to_grayscale(self.unnormalizeTensor(images_transformed)))[0])
            structure_loss *= self.lambda_structure
            del sobel_img, laplacian_img, canny_img

        if self.use_patchmatchnet:
            # print("Running PatchMatchNet for depth estimation")
            imgs.squeeze_(0)
            for i in range(imgs.shape[0]):
                imgs[i] = self.unnormalizeTensor(imgs[i])  # back to 0-1 range

            imgs = torch.split(imgs, 1, dim=0)  # list of B with 1x3xHxW
            images_transformed_norm = torch.zeros(images_transformed.shape)
            if self.lambda_volume > 0 or self.lambda_depth > 0:
                for i in range(images_transformed_norm.shape[0]):
                    images_transformed_norm[i] = self.unnormalizeTensor(images_transformed[i])
                    #images_transformed_norm[i] = torch.clamp(images_transformed_norm[i], 0, 1).clone()  # back to 0-1 range
                images_transformed_norm = images_transformed_norm.cuda()
                images_transformed_norm = torch.split(images_transformed_norm, 1, dim=0)
                depth = self.mvsnet(imgs, intrinsics.cuda(),
                                            extrinsics.cuda(), torch.Tensor([depth_min]).cuda(),
                                            torch.Tensor([depth_max]).cuda())
                depth_t = self.mvsnet(images_transformed_norm,
                                                intrinsics.cuda(),
                                                extrinsics.cuda(), torch.Tensor([depth_min]).cuda(),
                                                torch.Tensor([depth_max]).cuda())
                for l in range(len(depth)):
                    # volume_loss += self.smooth_l1_loss(volume[l][0], volume_t[l][0]) * 2 ** (1 - l)
                    depth_loss += self.smooth_l1_loss(depth[l][0], depth_t[l][0]) * 2 ** (1 - l)
                volume_loss *= self.lambda_volume
                depth_loss *= self.lambda_depth

                #del depth, volume, depth_t, volume_t, images_transformed_norm
                del depth, depth_t, images_transformed_norm


        else:
            # print("Running Cascade MVSNet for volume/depth loss")
            proj_mats.squeeze_(0)
            if self.lambda_volume > 0 or self.lambda_depth > 0:
                images_transformed = images_transformed.unsqueeze(0)
                volumes_original = self.mvsnet(imgs.cuda(), proj_mats.unsqueeze(0).cuda(), init_depth_min,
                                               depth_interval)
                volumes_transformed = self.mvsnet(images_transformed.cuda(), proj_mats.unsqueeze(0).cuda(),
                                                  init_depth_min, depth_interval)

                for l in range(3):
                    if self.lambda_volume > 0:
                        gt_pred_vol_l = volumes_original[f'prob_volume_{l}']
                        pred_vol_l = volumes_transformed[f'prob_volume_{l}']
                        volume_loss += self.smooth_l1_loss(gt_pred_vol_l, pred_vol_l) * 2 ** (1 - l)
                        del gt_pred_vol_l, pred_vol_l
                    if (self.lambda_depth > 0):
                        gt_pred_l = volumes_original[f'depth_{l}']
                        pred_l = volumes_transformed[f'depth_{l}']
                        depth_loss += self.smooth_l1_loss(pred_l, gt_pred_l) * 2 ** (1 - l)
                        del gt_pred_l, pred_l

                volume_loss = self.lambda_volume * volume_loss
                depth_loss = self.lambda_depth * depth_loss
                del volumes_original, volumes_transformed

                # Compute nnfm loss
        del imgs, images_transformed, proj_mats
        # torch.cuda.empty_cache()
        total_loss = content_loss + style_loss + volume_loss + depth_loss + structure_loss + nnfm_loss
        loss_log = {'total_loss': total_loss, 'style_loss': style_loss.item(), 'content_loss': content_loss.item(),
                    'volume_loss': volume_loss.item(),
                    'depth_loss': depth_loss.item(), 'nnfm_loss': nnfm_loss.item(),
                    'structure_loss': structure_loss.item()
                    }

        del volume_loss, depth_loss, nnfm_loss, style_loss, content_loss, structure_loss

        return loss_log

    def configure_optimizers(self):
        """ Configure optimizers for training """
        self.optimizer = None
        scheduler = None
        if self.hp.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hp.lr)
        if self.hp.lr_scheduler == 'multisteplr':
            scheduler = MultiStepLR(self.optimizer, milestones=[10, 30, 60, 80, 100], gamma=0.3)
        return [self.optimizer], [scheduler]

    def get_color_transfer_matrix(self, step=1):
        print("Computing color transfer matrix...")
        # collect all the images
        T = None
        if self.use_patchmatchnet:
            T = self.train_dataset[0]['images'][0]
            T = torch.from_numpy(T)
        else:
            T = self.unnormalizeTensor(self.train_dataset[0]['images'][0])

        T = T.permute(1, 2, 0).unsqueeze(0)

        for i in range(1, len(self.train_dataset), step):
            if self.use_patchmatchnet:
                tmp_img = self.train_dataset[i]['images'][0]
                tmp_img = torch.from_numpy(tmp_img)
            else:
                tmp_img = self.unnormalizeTensor(self.train_dataset[i]['images'][0])
            tmp_img = tmp_img.permute(1, 2, 0).unsqueeze(0)
            T = torch.cat((T, tmp_img), 0)
        st_img_orig = transforms.ToTensor()(Image.open('data/styles/%s.jpg' % (self.style_name)))
        st_img_orig = st_img_orig.permute(1, 2, 0)
        _, ct_mat = match_colors_for_image_set(T, st_img_orig)
        print("Color transfer matrix computed.")
        return ct_mat

    def training_step(self, batch, batch_nb):
        """ Training step """
        log = {'lr': self.lr}
        output = None
        if self.use_patchmatchnet:
            imgs, intrinsics, extrinsics, depth_min, depth_max = self.decode_batch(batch)
            output = self(imgs, proj_mats=None, init_depth_min=None, depth_interval=None,
                          intrinsics=intrinsics, extrinsics=extrinsics,
                          depth_min=depth_min, depth_max=depth_max)

        else:
            imgs, proj_mats, init_depth_min, depth_interval = self.decode_batch(batch)
            output = self(imgs, proj_mats, init_depth_min,
                          depth_interval)

            del proj_mats, init_depth_min, depth_interval, imgs,

        log['train/total_loss'] = total_loss = output['total_loss']
        log['train/style_loss'] = output['style_loss']
        log['train/content_loss'] = output['content_loss']
        log['train/volume_loss'] = output['volume_loss']
        log['train/depth_loss'] = output['depth_loss']
        log['train/nnfm_loss'] = output['nnfm_loss']
        log['train/structure_loss'] = output['structure_loss']
        tensorboard = self.logger.experiment
        tensorboard.add_scalar('train/total_loss', total_loss, self.global_step)
        tensorboard.add_scalar('train/style_loss', output['style_loss'], self.global_step)
        tensorboard.add_scalar('train/content_loss', output['content_loss'], self.global_step)
        tensorboard.add_scalar('train/volume_loss', output['volume_loss'], self.global_step)
        tensorboard.add_scalar('train/depth_loss', output['depth_loss'], self.global_step)
        tensorboard.add_scalar('train/nnfm_loss', output['nnfm_loss'], self.global_step)
        tensorboard.add_scalar('train/structure_loss', output['structure_loss'], self.global_step)

        my_log = {'loss': total_loss, 'style_loss': output['style_loss'], 'content_loss': output['content_loss'],
                  'volume_loss': output['volume_loss'], 'depth_loss': output['depth_loss'],
                  'nnfm_loss': output['nnfm_loss'], 'structure_loss': output['structure_loss'],
                  'log': log}
        del output
        return my_log

    def validation_step(self, batch, batch_nb):
        """ Validation step """
        log = {}
        output = None
        if self.use_patchmatchnet:
            imgs, intrinsics, extrinsics, depth_min, depth_max = self.decode_batch(batch)
            output = self(imgs, proj_mats=None, init_depth_min=None, depth_interval=None,
                          intrinsics=intrinsics, extrinsics=extrinsics,
                          depth_min=depth_min, depth_max=depth_max)
        else:
            imgs, proj_mats, init_depth_min, depth_interval = self.decode_batch(batch)
            output = self(imgs, proj_mats, init_depth_min,
                          depth_interval)
            del proj_mats
        log['val/total_loss'] = total_loss = output['total_loss'].item()
        log['val/style_loss'] = output['style_loss']
        log['val/content_loss'] = output['content_loss']
        log['val/volume_loss'] = output['volume_loss']
        log['val/depth_loss'] = output['depth_loss']
        log['val/nnfm_loss'] = output['nnfm_loss']
        log['val/structure_loss'] = output['structure_loss']

        image_samples = []

        if self.use_patchmatchnet:
            # Dataloader of PatchmatchNet returns a list of tensors
            # PatchmatchNet trained on 0-1 range images
            imgs = torch.stack(imgs, dim=1).squeeze(0)  # list to tensor conversion V, 3, H, W
            for i in range(imgs.shape[0]):
                imgs[i] = self.normalizeImg(transforms.ToPILImage()(imgs[i]))  # normalizing the image
            imgs = imgs.unsqueeze(0)  # add batch dimension

        for vid in range(self.n_views):
            image_samples += [
                self.image_transform(self.image_size)(self.unnormalizeTensor2Img(imgs.squeeze(0)[vid].cpu()))]
        image_samples = torch.stack(image_samples).cuda()

        if batch_nb == 0:
            if self.use_adain:
                output_img = self.adain_net(image_samples, self.style.cuda())
            else:
                output_img = self.transformer(image_samples)
            gt_image, out_image = image_samples.detach().clone().cpu(), output_img.detach().clone().cpu()
            image_grid = self.unnormalizeTensor(torch.cat((gt_image, out_image), 2))
            image_grid[image_grid < 0] = 0.0
            image_grid[image_grid > 1.] = 1.0
            log['val/image_grid'] = image_grid
            del image_grid, gt_image, out_image, output_img
        tensorboard = self.logger.experiment
        tensorboard.add_scalar('val/total_loss', total_loss, self.global_step)
        tensorboard.add_scalar('val/style_loss', output['style_loss'], self.global_step)
        tensorboard.add_scalar('val/content_loss', output['content_loss'], self.global_step)
        tensorboard.add_scalar('val/volume_loss', output['volume_loss'], self.global_step)
        tensorboard.add_scalar('val/depth_loss', output['depth_loss'], self.global_step)
        tensorboard.add_scalar('val/nnfm_loss', output['nnfm_loss'], self.global_step)
        tensorboard.add_scalar('val/structure_loss', output['structure_loss'], self.global_step)

        del imgs, image_samples, output
        torch.cuda.empty_cache()
        return log

    def validation_epoch_end(self, outputs):
        """ Validation epoch end """
        image_grid = outputs[0]['val/image_grid']
        mean_total_loss = torch.FloatTensor([x['val/total_loss'] for x in outputs]).mean()
        mean_style_loss = torch.FloatTensor([x['val/style_loss'] for x in outputs]).mean()
        mean_content_loss = torch.FloatTensor([x['val/content_loss'] for x in outputs]).mean()
        mean_volume_loss = torch.FloatTensor([x['val/volume_loss'] for x in outputs]).mean()
        mean_depth_loss = torch.FloatTensor([x['val/depth_loss'] for x in outputs]).mean()
        mean_nnfm_loss = torch.FloatTensor([x['val/nnfm_loss'] for x in outputs]).mean()
        mean_structure_loss = torch.FloatTensor([x['val/structure_loss'] for x in outputs]).mean()
        mvs = "patchmatchnet" if self.use_patchmatchnet else "casmvsnet"
        transfernet = "adain" if self.use_adain else "unet"
        save_image(image_grid,
                   "%s/%s_%d_%s_%s%s.png" % (
                       self.output_dir, self.style_name, self.current_epoch, mvs, transfernet, self.ablation_suffix))

        tensorboard = self.logger.experiment
        tensorboard.add_images('val/image_grid', image_grid, self.current_epoch)
        del image_grid
        return {'progress_bar': {'val_loss': mean_total_loss},
                'log': {'val/mean_total_loss': mean_total_loss,
                        'val/mean_style_loss': mean_style_loss,
                        'val/mean_content_loss': mean_content_loss,
                        'val/mean_volume_loss': mean_volume_loss,
                        'val/mean_depth_loss': mean_depth_loss,
                        'val/mean_nnfm_loss': mean_nnfm_loss,
                        'val/mean_structure_loss': mean_structure_loss
                        }
                }

    def training_epoch_end(self, outputs):
        """ Training epoch end """
        mean_total_loss = torch.stack([x['loss'] for x in outputs]).mean()
        mean_style_loss = torch.FloatTensor([x['style_loss'] for x in outputs]).mean()
        mean_content_loss = torch.FloatTensor([x['content_loss'] for x in outputs]).mean()
        mean_volume_loss = torch.FloatTensor([x['volume_loss'] for x in outputs]).mean()
        mean_depth_loss = torch.FloatTensor([x['depth_loss'] for x in outputs]).mean()
        mean_nnfm_loss = torch.FloatTensor([x['nnfm_loss'] for x in outputs]).mean()
        mean_structure_loss = torch.FloatTensor([x['structure_loss'] for x in outputs]).mean()

        print(
            'Training Epoch %d: mean_total_loss: %f, mean_style_loss: %f, mean_content_loss: %f, mean_volume_loss: %f, mean_depth_loss: %f, mean_nnfm_loss: %f, mean_structure_loss: %f' % (
                self.current_epoch, mean_total_loss, mean_style_loss, mean_content_loss, mean_volume_loss,
                mean_depth_loss, mean_nnfm_loss,
                mean_structure_loss
            ))
        

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    hparams = get_opts()
    print_args(hparams)

    torch.cuda.empty_cache()
    checkpoint_callback = ModelCheckpoint(filename=os.path.join(f'ckpt'
                                                                f's/{hparams.exp_name}',
                                                                '{epoch:02d}'),
                                          monitor='val/total_loss',
                                          mode='max',
                                          save_top_k=5, )

    logger = TensorBoardLogger(
        save_dir="logs",
        name=hparams.exp_name
    )
    system = StyleSystem(hparams)
    logger.log_hyperparams(hparams)
    torch.backends.cudnn.benchmark = True
    # find_unused_parameters=True
    from pytorch_lightning.strategies import DDPStrategy

    trainer = Trainer(max_epochs=system.epochs,
                      logger=logger,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPStrategy(find_unused_parameters=False),
                      num_sanity_val_steps=0 if hparams.num_gpus > 1 else 0,
                      benchmark=True,
                      precision=16 if False else 32)

    trainer.fit(system)
    
    inference_wh = system.image_size

    if system.use_patchmatchnet:
        train_dataset = MVSDataset(data_path=system.root_dir,
                                       num_views=system.n_views - 1,
                                       scan_name=system.scan_name,
                                       num_light_idx=-1,
                                       image_extension=system.image_extension,
                                       img_wh=inference_wh)
    else:
        train_dataset = CustomDataset(system.root_dir, '', True, n_views=system.n_views,
                                          img_wh=inference_wh, scan_name=system.scan_name, extension=system.image_extension)
    import datetime

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    transformToPIL = transforms.ToPILImage()

    st_img_orig = transforms.ToTensor()(Image.open('data/styles/%s.jpg' % (system.style_name)))
    st_img = transforms.Normalize(system.mean, system.std)(st_img_orig)
    st_img_norm = st_img.permute(1, 2, 0)
    st_img_orig = st_img_orig.permute(1, 2, 0)

    system.eval()
    with torch.no_grad():
        if system.post_color_transfer:
            # collect all the images
            print("Computing color transfer for stylized images")
        all_images = []
        for idx in range(0, len(train_dataset), system.color_steps):
            imgs = train_dataset[idx]['images']
            if system.use_patchmatchnet:
                # Read image shape
                imgs = [torch.from_numpy(numpy_array) for numpy_array in imgs]  # convert
                # import pdb;pdb.set_trace()
                if system.pre_color_transfer:
                    # "Torch scack the list of tensors patchmatchnet dataloader"
                    imgs = torch.stack(imgs, dim=0).permute(0, 2, 3, 1)  # [V, H, W, C]
                    # "Pre color transfer applied to patchmatchnet dataloader"
                    imgs = apply_color_tf(imgs, system.ct_mat).permute(0, 3, 1, 2)  # [V, C, H, W]
                    # "Image normalization applied to patchmatchnet dataloader"
                    imgs = system.normalizeTensor(imgs[0]).unsqueeze(0)  # normalizing the ref image, unsqueeze to add batch dim [1, C, H, W]
                else:
                    imgs = system.normalizeTensor(torch.stack(imgs, dim=0)[0]).unsqueeze(0)  # normalizing the ref image, unsqueeze to add batch dim [1, C, H, W]

            else:  # casMVSNet
                if system.pre_color_transfer:
                    imgs = apply_color_tf(system.unnormalizeTensor(imgs).squeeze(0).permute(0, 2, 3, 1), system.ct_mat)
                    imgs = system.normalizeTensor(imgs.permute(0, 3, 1, 2))[0].unsqueeze(
                        0)  # normalizing the ref image, unsqueeze to add batch dim [1, C, H, W]
            style_tf = system.style
            if system.use_adain:
                alpha = 1.
                images_transformed, t = system.adain_net(imgs, system.style, 1., True)
                image_grid = system.unnormalizeTensor(images_transformed)[0]
            else:  # UNET
                transformer = system.transformer
                imgs = system.image_transform(system.image_size)(
                    system.unnormalizeTensor2Img(torch.Tensor(imgs[0])))
                images_transformed = transformer(imgs.unsqueeze(0))
                image_grid = system.unnormalizeTensor(images_transformed)[0]
            # image grid is [C, H, W]
            image_grid[image_grid < 0] = 0.0
            image_grid[image_grid > 1.] = 1.0
            if system.post_color_transfer:
                all_images.append(image_grid.permute(1, 2, 0).unsqueeze(0))
                progress_bar(idx + 1, len(train_dataset), "Post color transfer")
            else:
                if (idx % system.color_steps == 0):
                    save_image(image_grid, f'%s/%08d%s.png' % (
                        system.output_dir, idx // system.color_steps, system.ablation_suffix))
                    progress_bar(idx + 1, len(train_dataset), "Saving images")
        if system.post_color_transfer:
            all_images = torch.cat(all_images, dim=0)
            for jdx in range(0, len(train_dataset), system.color_steps):
                save_image(all_images[jdx // system.color_steps].permute(2, 0, 1),
                           f'%s/%08d%s.png' % (system.output_dir, jdx // system.color_steps, system.ablation_suffix))
            all_images, _ = match_colors_for_image_set(all_images, st_img_orig)
            for jdx in range(0, len(train_dataset), system.color_steps):
                save_image(all_images[jdx // system.color_steps].permute(2, 0, 1),
                           f'%s/%08d%s_ct.png' % (system.output_dir, jdx // system.color_steps, system.ablation_suffix))
