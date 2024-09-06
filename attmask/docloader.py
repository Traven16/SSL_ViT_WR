from typing import Any
import numpy as np
import itertools
from multiprocessing import Pool
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


variable = 51

class icdar2017patcher(Dataset):
    def __init__(self, data_path, window_size=256, overlap=0.5, at_least_white=0.0, multiprocessing=True, transform=None, target_transform=None):
        print('Using ICDAR2017 Patcher')
        self.data_path = data_path
        self.files = Path(self.data_path).glob('*')
        self.docs = { file.name: {'img': Image.open(str(file)).convert('L'), 'writer_id': int(str(file.name).split('-')[0]), 'page_id': i} for i, file in enumerate(self.files)}
        self.window_size = window_size
        self.overlap = overlap
        self.at_least_white = at_least_white

        self.transform = transform
        self.target_transform = target_transform

        print('Validating patches')
        # list of tules (filename, (center_y, center_x))

        if not multiprocessing:
            self.patch_coords = []
            for idx, fname in enumerate(self.docs.keys()):
                print(f'[{idx}/{len(list(self.docs))}] processed documents', end='\r')

                self.patch_coords += __extract_patches__(fname, self.docs[fname], self.window_size, self.overlap, self.at_least_white)

        else:
            param_list = [(fname, fdict, self.window_size, self.overlap, self.at_least_white) for fname, fdict in self.docs.items()]
            with Pool() as pool:
                self.patch_coords = list(itertools.chain(*pool.starmap(__extract_patches__, param_list)))

    def __len__(self) -> int:
        return len(self.patch_coords)

    def __getitem__(self, index) -> Any:
        fname, (y_coord, x_coord) = self.patch_coords[index]
        doc = self.docs[fname]
        img = doc['img'].convert('RGB')
        writer = doc['writer_id']
        page = doc['page_id']

        patch = __crop__(img, y_coord, x_coord, self.window_size)
        
        if self.transform is not None:
            patch = self.transform(patch)

        if self.target_transform is not None:
            writer = self.target_transform(writer)

        return patch, writer, page
    
def __extract_patches__(fname, fdict, window_size, overlap, at_least_white):
    valid_coords = []
    img = fdict['img'] #.convert('RGB')
    width, height = img.size
    stride = int(window_size - window_size * overlap) # overlap in ratio <0

    fdict['height'] = height
    fdict['width'] = width

    assert window_size <= height and window_size <= width, "Patch size is bigger than image size!"

    # starting coords for each window
    ws = window_size
    x_coords = list(range(0, width - ws, stride))
    y_coords = list(range(0, height - ws, stride))

    # x_residual = x_coords[-1] + window_size
    # y_residual = y_coords[-1] + window_size

    # # shift coords by half of remainder
    # x_coords = np.array(x_coords) + (int(x_residual//2))
    # y_coords = np.array(y_coords) + (int(y_residual//2))

    fdict['num_x'] = len(x_coords)
    fdict['num_y'] = len(y_coords)

    existing_coords = list(itertools.product(y_coords, x_coords))
    existing_coords = list(zip([fname] * len(existing_coords), existing_coords))

    if at_least_white > 0:
        for fname, coord in existing_coords:
            patch = np.array(__crop__(img, *coord, window_size))
            non_black_px_idx = np.any(patch != [0,0,0], axis=-1)
            non_black_px_ratio = non_black_px_idx.sum() / (patch.shape[0] * patch.shape[1])

            if not non_black_px_ratio >= at_least_white:
                continue

            valid_coords.append((fname, coord))
    else:
        valid_coords = existing_coords

    return valid_coords

def __crop__(img, y_coord, x_coord, window_size):
    return img.crop((
        x_coord,
        y_coord,
        x_coord + (window_size),
        y_coord + (window_size)))
    
