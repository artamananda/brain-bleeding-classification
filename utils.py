import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import transforms
from PIL import Image
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import io
import base64

def run_detection():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")

    print(device)

    #Dataloader
    class MaskDataset(object):
        def __init__(self, transforms, path):
            '''
            path: path to train folder or test folder
            '''
            # transform module과 img path 경로를 정의
            self.transforms = transforms
            self.path = path
            self.imgs = list(sorted(os.listdir(self.path)))


        def __getitem__(self, idx): #special method
            # load images ad masks
            file_image = self.imgs[idx]
            img_path = os.path.join(self.path, file_image)


            img = Image.open(img_path).convert("RGB")

            if self.transforms is not None:
                img = self.transforms(img)

            return img

        def __len__(self):
            return len(self.imgs)

    data_transform = transforms.Compose([ 
            transforms.ToTensor()
        ])

    def collate_fn(batch):
        return torch.stack(batch) 

    test_dataset = MaskDataset(data_transform, 'static/uploads/')

    test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    def get_model_instance_segmentation(num_classes):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    model = get_model_instance_segmentation(2)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.load_state_dict(torch.load('model/model_100_0.001_4_ResNet50_split9010.pt', map_location=torch.device('cpu')))


    def plot_image_from_output(img, annotation):
        detectionStatus = False
        img = img.cpu().permute(1,2,0)

        _,ax = plt.subplots(1)
        ax.imshow(img)

        for idx in range(len(annotation["boxes"])):
            xmin, ymin, xmax, ymax = annotation["boxes"][idx]

            if annotation['labels'][idx] == 1:
                rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='r', facecolor='none')
                label = "True"
                detectionStatus = True
            else:
                rect = patches.Rectangle((xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=1, edgecolor='orange', facecolor='none')
                label = "False"

            ax.add_patch(rect)

            # Menentukan koordinat untuk teks di luar kotak
            text_x = xmax + 5
            text_y = (ymin + ymax) / 2

            # Menambahkan label teks di luar kotak
            score = annotation['scores'][idx] if 'scores' in annotation else None
            text = f"{label} {int(score * 100)}%" if score is not None else label
            ax.text(text_x, text_y, text, fontsize=10, color='white', verticalalignment='center', bbox={'color': 'black', 'alpha': 0.7, 'pad': 0})

        # plt.show()
        imgPng = io.BytesIO()
        plt.savefig(imgPng, format='png')
        imgPng.seek(0)
        plot_url = base64.b64encode(imgPng.getvalue()).decode()
        return plot_url, detectionStatus

    def make_prediction(model, img, threshold):
        model.eval()
        preds = model(img)
        for id in range(len(preds)) :
            idx_list = []

            for idx, score in enumerate(preds[id]['scores']) :
                if score > threshold :
                    idx_list.append(idx)

            preds[id]['boxes'] = preds[id]['boxes'][idx_list]
            preds[id]['labels'] = preds[id]['labels'][idx_list]
            preds[id]['scores'] = preds[id]['scores'][idx_list]

        return preds

    with torch.no_grad():
        for imgs in test_data_loader:
            imgs = list(img.to(device) for img in imgs)
            pred = make_prediction(model, imgs, 0.3)


    _idx = 0
    print("Prediction : ", pred[_idx]['labels'])
    print(pred[_idx])

    # Create a tensor on the CUDA device
    for key in pred[_idx]:
        pred[_idx][key] = pred[_idx][key].cpu()

    plot_urls, detectionStatusAll = plot_image_from_output(imgs[_idx], pred[_idx])

    return plot_urls, detectionStatusAll

def plot_image(image_path):
    img = Image.open(image_path)

    fig,ax = plt.subplots(1)
    ax.imshow(img, cmap='gray')

    imgPng = io.BytesIO()
    plt.savefig(imgPng, format='png')
    imgPng.seek(0)
    plot_url = base64.b64encode(imgPng.getvalue()).decode()
    return plot_url

def delete_file(filepath):
    try:
        for filename in os.listdir(filepath):
            file_path = os.path.join(filepath, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print("Error deleting file:", e)

def checking_file_format(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS