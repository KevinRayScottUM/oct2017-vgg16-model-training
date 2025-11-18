import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
from PIL import Image  # 需要导入PIL库来处理图像

# 定义训练好的模型文件路径
model_path = 'VGG16_OCT_Retina_trained_model.pt'
# Classes:
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# 定义输入图像的转换操作
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# 创建一个函数，用于加载和预处理单个图像以进行推理
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 确保图像以RGB格式打开
    image = data_transforms(image)
    image = Variable(image.unsqueeze(0))
    return image

# 载入VGG16模型
vgg16 = models.vgg16_bn()
vgg16.classifier[-1] = nn.Linear(4096, 4)  # 根据你的具体问题调整输出层
vgg16.load_state_dict(torch.load(model_path))
vgg16.eval()

# 设置设备为GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg16 = vgg16.to(device)

# 测试图像的目录路径
test_images_dir = 'test/'

# 获取测试目录中所有图像文件的列表
test_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith('.jpeg')]

# 存储所有图像和预测类别的列表
all_images = []
all_predicted_classes = []

# 遍历每个测试图像，执行推理
for img_path in test_images:
    # 处理图像
    input_image = process_image(img_path).to(device)

    # 进行推理
    with torch.no_grad():
        output = vgg16(input_image)

    # 获取预测的类别索引
    _, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    # 存储图像和预测类别
    all_images.append(Image.open(img_path))
    all_predicted_classes.append(predicted_class)

# 显示所有图像在一个面板下
fig, axes = plt.subplots(1, len(test_images), figsize=(20, 5))

for i, (img, predicted_class) in enumerate(zip(all_images, all_predicted_classes)):
    axes[i].imshow(img)
    axes[i].set_title(f'Predicted Class: {class_names[predicted_class]}')
    axes[i].axis('off')

plt.show()
