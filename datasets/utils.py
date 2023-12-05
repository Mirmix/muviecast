import numpy as np
import re
import sys
import torchvision.utils
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Callable, Union, Dict


def print_args(args: Any) -> None:
    """Utilities to print arguments

    Args:
        args: arguments to print out
    """
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


def make_nograd_func(func: Callable) -> Callable:
    """Utilities to make function no gradient

    Args:
        func: input function

    Returns:
        no gradient function wrapper for input function
    """

    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


def make_recursive_func(func: Callable) -> Callable:
    """Convert a function into recursive style to handle nested dict/list/tuple variables

    Args:
        func: input function

    Returns:
        recursive style function
    """

    def wrapper(args):
        if isinstance(args, list):
            return [wrapper(x) for x in args]
        elif isinstance(args, tuple):
            return tuple([wrapper(x) for x in args])
        elif isinstance(args, dict):
            return {k: wrapper(v) for k, v in args.items()}
        else:
            return func(args)

    return wrapper


@make_recursive_func
def tensor2float(args: Any) -> float:
    """Convert tensor to float"""
    if isinstance(args, float):
        return args
    elif isinstance(args, torch.Tensor):
        return args.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(args)))


@make_recursive_func
def tensor2numpy(args: Any) -> np.ndarray:
    """Convert tensor to numpy array"""
    if isinstance(args, np.ndarray):
        return args
    elif isinstance(args, torch.Tensor):
        return args.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(args)))


@make_recursive_func
def to_cuda(args: Any) -> Union[str, torch.Tensor]:
    """Convert tensor to tensor on GPU"""
    if isinstance(args, torch.Tensor):
        return args.cuda()
    elif isinstance(args, str):
        return args
    else:
        raise NotImplementedError("invalid input type {} for to_cuda".format(type(args)))


def save_scalars(logger: SummaryWriter, mode: str, scalar_dict: Dict[str, Any], global_step: int) -> None:
    """Log values stored in the scalar dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        scalar_dict: python dictionary stores the key and value pairs to be recorded
        global_step: step index where the logger should write
    """
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = "{}/{}".format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = "{}/{}_{}".format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger: SummaryWriter, mode: str, images: Dict[str, np.ndarray], global_step: int) -> None:
    """Log images stored in the image dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        images: python dictionary stores the key and image pairs to be recorded
        global_step: step index where the logger should write
    """
    def preprocess(image_name, image):
        if not (len(image.shape) == 3 or len(image.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(image_name, image.shape))
        if len(image.shape) == 3:
            image = image[:, np.newaxis, :, :]
        image = torch.from_numpy(image[:1])
        return torchvision.utils.make_grid(image, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images.items():
        if not isinstance(value, (list, tuple)):
            name = "{}/{}".format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = "{}/{}_{}".format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter:
    """Wrapper class for dictionary variables that require the average value"""

    def __init__(self) -> None:
        """Initialization method"""
        self.data: Dict[Any, float] = {}
        self.count = 0

    def update(self, new_input: Dict[Any, float]) -> None:
        """Update the stored dictionary with new input data

        Args:
            new_input: new data to update self.data
        """
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self) -> Any:
        """Return the average value of values stored in self.data"""
        return {k: v / self.count for k, v in self.data.items()}


def compute_metrics_for_each_image(metric_func: Callable) -> Callable:
    """A wrapper to compute metrics for each image individually"""

    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def threshold_metrics(
    depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Return error rate for where absolute error is larger than threshold.

    Args:
        depth_est: estimated depth map
        depth_gt: ground truth depth map
        mask: mask
        threshold: threshold

    Returns:
        error rate: error rate of the depth map
    """
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt).float()
    err_mask = errors > threshold
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def absolute_depth_error_metrics(depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate average absolute depth error

    Args:
        depth_est: estimated depth map
        depth_gt: ground truth depth map
        mask: mask
    """
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt).abs())
def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None
    
    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()