import torch
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
import pandas as pd
from PIL import Image

use_gpu= torch.cuda.is_available()
DATASET_ROOT = './'
PATH_TO_WEIGHTS = './model/model.pth'


def test():
    data_transform = transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset_root = Path(DATASET_ROOT)
    classes = [_dir.name for _dir in dataset_root.joinpath('train').glob('*')]

    model = torch.load(PATH_TO_WEIGHTS)
    if use_gpu:
        model = model.cuda(1)
    model.eval()

    sample_submission = pd.read_csv(str(dataset_root.joinpath('sample_submission.csv')))
    submission = sample_submission.copy()
    for i, filename in enumerate(sample_submission['file']):
        image = Image.open(str(dataset_root.joinpath('test').joinpath(filename))).convert('RGB')
        image = data_transform(image).unsqueeze(0)
        if use_gpu:
            inputs = Variable(image.cuda(1))
        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        submission['species'][i] = classes[preds[0]]

    submission.to_csv(str(dataset_root.joinpath('submission.csv')), index=False)
    


if __name__ == '__main__':
    test()