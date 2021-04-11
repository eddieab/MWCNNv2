from importlib import import_module
from data import hdrd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataloader import default_collate


class Data:
    def __init__(self, args):
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = self.loader_train = DataLoader(
                hdrd.HDRD(args),
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )

        self.loader_test = DataLoader(
            hdrd.HDRD(args, train=False),
            batch_size=1,
            shuffle=False,
            **kwargs
        )
