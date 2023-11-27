import numpy as np
import scipy
import torch
from torch.utils.data import DataLoader, Dataset


class PointCloud2dDataset(Dataset):
    def __init__(self, cfg, loadNovel=False, loadFixedOut=True, loadTest=False):
        self.cfg = cfg
        self.loadNovel = loadNovel
        self.loadFixedOut = loadFixedOut
        self.load = "test" if loadTest else "train"
        list_file = f"{cfg.path}/{cfg.category}_{self.load}.list"
        self.CADs = []
        with open(list_file) as file:
            for line in file:
                id = line.strip().split("/")[1]
                self.CADs.append(id)
            self.CADs.sort()

    def __len__(self):
        return len(self.CADs)

    def __getitem__(self, idx):
        CAD = self.CADs[idx]
        image_in = np.load(f"{self.cfg.path}/{self.cfg.category}_inputRGB/{CAD}.npy")
        image_in = image_in / 255.0

        point_cloud = np.load(f"{self.cfg.path}/{self.cfg.category}_pointCloud/{CAD}.npy")

        if self.loadNovel:
            raw_data = scipy.io.loadmat(f"{self.cfg.path}/{self.cfg.category}_depth/{CAD}.mat")
            depth = raw_data["Z"]
            trans = raw_data["trans"]
            mask = depth != 0
            depth[~mask] = self.cfg.renderDepth

            return {"image_in": image_in, "point_cloud": point_cloud, "depth": depth, "mask": mask, "trans": trans}

        if self.loadFixedOut:
            raw_data = scipy.io.loadmat(f"{self.cfg.path}/{self.cfg.category}_depth_fixed{self.cfg.outViewN}/{CAD}.mat")
            depth = raw_data["Z"]
            mask = depth != 0
            depth[~mask] = self.cfg.renderDepth

            return {"image_in": image_in, "point_cloud": point_cloud, "depth": depth, "mask": mask}

    def collate_fn(self, batch):
        batch_n = {key: np.array([d[key] for d in batch]) for key in batch[0]}
        modelIdx = np.random.permutation(self.cfg.chunkSize)[:self.cfg.batchSize]
        modelIdxTile = np.tile(modelIdx, [self.cfg.novelN, 1]).T
        angleIdx = np.random.randint(self.cfg.inputViewN, size=[self.cfg.batchSize])
        sampleIdx = np.random.randint(self.cfg.sampleN, size=[self.cfg.batchSize, self.cfg.novelN])

        images = batch_n["image_in"][modelIdx, angleIdx]
        point_clouds = batch_n["point_cloud"][modelIdx]
        targetTrans = batch_n["trans"][modelIdxTile, sampleIdx]
        depthGT = np.expand_dims(batch_n["depth"][modelIdxTile, sampleIdx], axis=-1)
        maskGT = np.expand_dims(batch_n["mask"][modelIdxTile, sampleIdx], axis=-1).astype(np.int)

        images = torch.from_numpy(images).permute((0, 3, 1, 2))
        point_clouds = torch.from_numpy(point_clouds).float()
        targetTrans = torch.from_numpy(targetTrans)
        depthGT = torch.from_numpy(depthGT).permute((0, 1, 4, 2, 3))
        maskGT = torch.from_numpy(maskGT).permute((0, 1, 4, 2, 3))

        return {"inputImage": images, "pointCloud": point_clouds, "targetTrans": targetTrans, "depthGT": depthGT, "maskGT": maskGT}

    def collate_fn_fixed(self, batch):
        batch_n = {key: np.array([d[key] for d in batch]) for key in batch[0]}
        modelIdx = np.random.permutation(self.cfg.chunkSize)[:self.cfg.batchSize]
        angleIdx = np.random.randint(24, size=[self.cfg.batchSize])

        images = batch_n["image_in"][modelIdx, angleIdx]
        point_clouds = batch_n["point_cloud"][modelIdx]
        depthGT = np.transpose(batch_n["depth"][modelIdx], axes=[0, 2, 3, 1])
        maskGT = np.transpose(batch_n["mask"][modelIdx], axes=[0, 2, 3, 1]).astype(int)

        images = torch.from_numpy(images).permute((0, 3, 1, 2))
        point_clouds = torch.from_numpy(point_clouds).float()
        depthGT = torch.from_numpy(depthGT).permute((0, 3, 1, 2))
        maskGT = torch.from_numpy(maskGT).permute((0, 3, 1, 2))

        return {"inputImage": images, "pointCloud": point_clouds, "depthGT": depthGT, "maskGT": maskGT}


if __name__ == "__main__":
    import options

    CFG = options.get_arguments()

    ds_fixed = PointCloud2dDataset(CFG)
    dl_fixed = DataLoader(ds_fixed, batch_size=CFG.chunkSize, shuffle=False, collate_fn=ds_fixed.collate_fn_fixed, num_workers=0)

    ds_novel = PointCloud2dDataset(CFG, loadNovel=True)
    dl_novel = DataLoader(ds_novel, batch_size=CFG.chunkSize, shuffle=False, collate_fn=ds_novel.collate_fn, num_workers=0)
