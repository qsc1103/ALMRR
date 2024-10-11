import torch
import torch.nn.functional as F
from feature_resnet import Extractor_ResNet50
from torch.utils.data import DataLoader
import torch.optim as optim
from MVTec import MVTecTrainDataset, MVTecTestDataset
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score
import cv2
import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from model import MFRM, FRM
from loss import FocalLoss, DiceLoss
import logging
import os






class ALMRR():
    """
    Anomaly segmentation model: DFR.
    """
    def __init__(self, cfg):
        super(ALMRR, self).__init__()
        self.cfg = cfg
        self.path = cfg.save_path    # model and results saving path
        self.batch_size = cfg.batch_size
        self.data_name = cfg.data_name
        self.img_size = cfg.img_size
        self.device = torch.device(cfg.device)

        if cfg.backbone == 'resnet':
            self.extractor = Extractor_ResNet50(upsample=cfg.upsample,
                                                is_agg=cfg.is_agg,
                                                kernel_size=cfg.kernel_size,
                                                stride=cfg.stride,
                                                dilation=cfg.dilation,
                                                featmap_size=cfg.featmap_size,
                                                device=cfg.device).to(self.device)

        # datasets
        self.train_data_path = cfg.train_data_path + self.data_name + '/train/good/'
        self.anomaly_source_path = cfg.anomaly_source_path
        self.test_data_path = cfg.train_data_path + self.data_name + '/test/'
        self.train_data = MVTecTrainDataset(root_dir=self.train_data_path, anomaly_source_path=self.anomaly_source_path, normalize=True)
        self.test_data = MVTecTestDataset(root_dir=self.test_data_path, resize_shape=[256, 256])

        # dataloader
        self.train_data_loader = DataLoader(self.train_data, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        self.test_data_loader = DataLoader(self.test_data, batch_size=1, shuffle=False, num_workers=0)

        # autoencoder classifier
        self.autoencoder, self.discriminative_net, self.model_name = self.build_classifier()
        self.best_image_auroc = 0

        # 设置日志记录
        self.setup_logging()

        # saving paths
        self.subpath = self.data_name + "/" + self.model_name
        self.model_path = os.path.join(self.path, "models/" + self.subpath)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)


        # optimizer
        self.lr = cfg.lr
        self.optimizer = optim.Adam(
            list(self.autoencoder.parameters()) + list(self.discriminative_net.parameters()),
            lr=self.lr,
            weight_decay=0
        )

    def setup_logging(self):
        """
        Setup logging configuration
        """
        log_dir = os.path.join(self.path, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = os.path.join(log_dir, f'{self.data_name}_training.log')

        # 配置日志记录器
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 在日志中记录初始配置信息
        logging.info("Training Initialized")
        logging.info(f"Model Name: {self.model_name}")
        logging.info(f"Data Name: {self.data_name}")

    def build_classifier(self):
        autoencoder = MFRM(in_channels=1792, latent_dim=1024).to(self.device)
        discriminative_net = FRM(in_channels=2, out_channels=2).to(self.device)
        model_name = "ALMRR"
        return autoencoder, discriminative_net, model_name

    def optimize_step(self, nor_img, aug_img, anomaly_mask):
        self.extractor.eval()
        self.autoencoder.train()
        self.discriminative_net.train()

        self.optimizer.zero_grad()

        # forward
        input_data = self.extractor(aug_img)
        nor_data = self.extractor(nor_img)
        dec = self.autoencoder(input_data)
        dec_mean = torch.mean(dec, dim=1).unsqueeze(1)
        input_data_mean = torch.mean(input_data, dim=1).unsqueeze(1)
        joined_in = torch.cat((dec_mean, input_data_mean), dim=1)
        predict_mask = self.discriminative_net(joined_in)
        predict_mask_sm = torch.softmax(predict_mask, dim=1)
        predict_mask_sm = F.interpolate(predict_mask_sm, size=(256, 256), mode='bilinear', align_corners=True)

        # loss
        res_loss = self.autoencoder.loss_function(dec, nor_data)
        loss_focal = FocalLoss()
        seg_loss = loss_focal(predict_mask_sm, anomaly_mask)
        total_loss = res_loss + seg_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss

    def validation(self):
        """
        Perform validation on the test dataset using the current model's weights.
        Returns four metrics: Image AUROC, Pixel AUROC, Image AP, Pixel AP.
        """
        img_dim = 256
        self.extractor.eval()  # 设置为评估模式
        self.autoencoder.eval()  # 设置为评估模式
        self.discriminative_net.eval()  # 设置为评估模式

        anomaly_score_gt = []  # 用于存储每张图像的异常标签（0表示正常，1表示异常）
        anomaly_score_prediction = []  # 用于存储每张图像的预测分数

        total_gt_pixel_scores = np.zeros((img_dim * img_dim * len(self.test_data)))
        total_pixel_scores = np.zeros((img_dim * img_dim * len(self.test_data)))
        mask_cnt = 0

        with torch.no_grad():  # 在验证过程中不需要梯度计算
            for i_batch, sample_batched in enumerate(self.test_data_loader):
                gray_batch = sample_batched["image"].to(self.device)
                is_normal = sample_batched["has_anomaly"].detach().cpu().numpy()[0, 0]
                anomaly_score_gt.append(is_normal)

                true_mask = sample_batched["mask"].cpu().detach().numpy()[0, :, :, :].transpose((1, 2, 0))
                gray_batch = self.extractor(gray_batch)
                gray_rec = self.autoencoder(gray_batch)

                dec_mean = torch.mean(gray_rec, dim=1).unsqueeze(1)
                input_data_mean = torch.mean(gray_batch, dim=1).unsqueeze(1)
                joined_in = torch.cat((dec_mean, input_data_mean), dim=1)

                out_mask = self.discriminative_net(joined_in)
                out_mask = torch.softmax(out_mask, dim=1)
                out_mask = F.interpolate(out_mask, size=(img_dim, img_dim), mode='bilinear', align_corners=True)
                out_mask_cv = out_mask[0, 1, :, :].cpu().detach().numpy()

                # 计算每张图像的异常分数（最大值）
                out_mask_averaged = torch.nn.functional.avg_pool2d(out_mask[:, 1, :, :], 9, stride=1, padding=9 // 2).cpu().detach().numpy()
                image_score = np.max(out_mask_averaged)
                anomaly_score_prediction.append(image_score)

                # 展平的像素级别分数
                flat_true_mask = true_mask.flatten()
                flat_out_mask = out_mask_cv.flatten()
                total_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_out_mask
                total_gt_pixel_scores[mask_cnt * img_dim * img_dim:(mask_cnt + 1) * img_dim * img_dim] = flat_true_mask
                mask_cnt += 1

        # 计算Image级别和Pixel级别的AUROC和AP
        anomaly_score_prediction = np.array(anomaly_score_prediction)
        anomaly_score_gt = np.array(anomaly_score_gt).astype(int)
        
        # Image级别的AUROC和AP
        image_auroc = roc_auc_score(anomaly_score_gt, anomaly_score_prediction)
        image_ap = average_precision_score(anomaly_score_gt, anomaly_score_prediction)
        
        # Pixel级别的AUROC和AP
        total_gt_pixel_scores = total_gt_pixel_scores[:mask_cnt * img_dim * img_dim].astype(np.uint8)
        total_pixel_scores = total_pixel_scores[:mask_cnt * img_dim * img_dim]
        pixel_auroc = roc_auc_score(total_gt_pixel_scores, total_pixel_scores)
        pixel_ap = average_precision_score(total_gt_pixel_scores, total_pixel_scores)

        # 返回四个指标：Image AUROC, Pixel AUROC, Image AP, Pixel AP
        return image_auroc, pixel_auroc, image_ap, pixel_ap

    def save_model(self, epoch=None):
        """
        Save the model's state as 'best_model.pth'.
        The 'epoch' parameter is optional and used only for logging purposes.
        """
        filename = 'best_model.pth'
        torch.save({
            'autoencoder': self.autoencoder.state_dict(),
            'discriminativenet': self.discriminative_net.state_dict()
        }, os.path.join(self.model_path, filename))

        if epoch is not None:
            logging.info(f"Best model saved for epoch {epoch}")
        else:
            logging.info("Best model saved.")
            
        print(f"Model saved as 'best_model.pth' for epoch {epoch}")

            
    def load_model(self, epoch=None):
        """
        Load the saved model's weights from a specific epoch.
        If epoch is None, it will try to load the most recent or default model.
        """

        model_path = os.path.join(self.model_path, 'best_model.pth')  # 默认的模型文件名

        if not os.path.exists(model_path):
            print(f"Model weights not found at {model_path}")
            return False
    
        print(f"Loading model weights from {model_path}")
    
        # 加载模型权重到当前模型
        checkpoint = torch.load(model_path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint['autoencoder'])
        self.discriminative_net.load_state_dict(checkpoint['discriminativenet'])
    
        print("Model weights loaded successfully.")
        return False


    def train(self):
        if self.load_model():
            logging.info("Model Loaded.")
            return

        start_time = time.time()
        iters_per_epoch = len(self.train_data_loader)
        epochs = self.cfg.epochs
        self.best_image_auroc = 0

        for epoch in range(1, epochs + 1):
            self.extractor.train()
            self.autoencoder.train()
            self.discriminative_net.train()
            losses = []

            for i, sample_batched in enumerate(self.train_data_loader):
                nor_img = sample_batched['image'].to(self.device)
                aug_img = sample_batched['augmented_image'].to(self.device)
                anomaly_mask = sample_batched['anomaly_mask'].to(self.device)

                # 执行优化步骤并计算损失
                total_loss = self.optimize_step(nor_img, aug_img, anomaly_mask)
                losses.append(total_loss.item())

            # 每个epoch记录一次损失
            avg_loss = np.mean(losses)
            logging.info(f'Epoch {epoch}/{epochs} - Loss: {avg_loss}')
            print(f'Epoch {epoch}/{epochs} - Loss: {avg_loss}')

            # 每10个epoch进行一次验证并保存模型
            if epoch % 10 == 0:
                image_auroc, pixel_auroc, image_ap, pixel_ap = self.validation()  # 进行验证，得到四个指标

                # 打印并记录验证结果
                logging.info(f"Epoch {epoch} - Image AUROC: {image_auroc}, Pixel AUROC: {pixel_auroc}, Image AP: {image_ap}, Pixel AP: {pixel_ap}")
                print(f"Epoch {epoch} - Image AUROC: {image_auroc}, Pixel AUROC: {pixel_auroc}, Image AP: {image_ap}, Pixel AP: {pixel_ap}")

                # 如果Image AUROC比之前的最佳结果好，则更新最佳指标并保存模型
                if image_auroc > self.best_image_auroc:
                    self.best_image_auroc = image_auroc
                    self.save_model(epoch)
                    logging.info(f"New best model found at epoch {epoch} with Image AUROC: {image_auroc}")
                    print(f"New best model found at epoch {epoch} with Image AUROC: {image_auroc}")

        total_time = time.time() - start_time
        logging.info(f"Training complete. Best Image AUROC: {self.best_image_auroc}")
        logging.info(f"Total training time: {str(datetime.timedelta(seconds=total_time))}")
        print(f"Training complete. Best Image AUROC: {self.best_image_auroc}")
        print(f"Total training time: {str(datetime.timedelta(seconds=total_time))}")

