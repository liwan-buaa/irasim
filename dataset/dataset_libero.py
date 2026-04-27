from dataset.dataset_multiview_action import MultiViewActionDataset


class Dataset_Libero(MultiViewActionDataset):
    def __init__(self, args, mode="val"):
        super().__init__(args=args, mode=mode, dataset_name="libero", action_dim=7)
