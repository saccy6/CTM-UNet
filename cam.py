import cv2
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的ResNet-50模型
model = models.resnet50(pretrained=False)
checkpoint = torch.load('D:/电脑文档/跑过的code/UNeXt-400epoch/models/model_ISIC2018_UNext_woDS.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 替换目标层为全局平均池化层
model.layer4[-1] = nn.AdaptiveAvgPool2d(1)

# 设置目标层（用于Grad-CAM的全局平均池化层）
target_layer = model.layer4[-1]


# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载图像并进行预处理
def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# 使用Grad-CAM可视化图像区域重要性
def visualize_grad_cam(model, img_tensor, target_layer, class_index):
    def backward_hook(module, grad_in, grad_out):
        global gradients
        gradients = grad_in[0]

    # 注册backward hook
    target_layer.register_backward_hook(backward_hook)

    # 前向传播
    output = model(img_tensor)
    target_class_output = output[0, class_index]

    # 反向传播并计算梯度
    model.zero_grad()
    target_class_output.backward()

    # 计算Grad-CAM权重
    pooled_gradients = torch.mean(gradients, dim=[2, 3])
    activations = target_layer(img_tensor).detach()
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[:, i]

    # 计算类激活图
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    # 将类激活图转换为numpy数组
    heatmap = heatmap.cpu().numpy()

    return heatmap

if __name__ == "__main__":
    image_path = r"C:/Users/84743/Desktop/0000008.png"  # 替换为你的图像路径

    # 加载图像并进行预处理
    img_tensor = load_and_preprocess_image(image_path)

    # 使用ResNet-50进行预测
    with torch.no_grad():
        output = model(img_tensor)

    # 获取最可能的类别索引
    _, class_index = torch.max(output, 1)
    class_index = class_index.item()

    # 使用Grad-CAM可视化图像区域重要性
    heatmap = visualize_grad_cam(model, img_tensor, target_layer, class_index)

    # 可视化原始图像和Grad-CAM图像
    img = Image.open(image_path).convert('RGB')


    # 将热力图与原始图像叠加
    def overlay_heatmap(img, heatmap, alpha=0.5):
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        img = np.float32(img) / 255
        cam = heatmap + img
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)

        return cam
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
#
# # 使用ResNet-50作为编码器
# model = tf.keras.applications.ResNet50(weights='imagenet', include_top=True)
#
# # 加载图像并进行预处理
# def load_and_preprocess_image(image_path):
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array = preprocess_input(img_array)
#     return img_array
#
# # 使用Grad-CAM可视化图像区域重要性
# def visualize_grad_cam(model, img_array, class_index, layer_name):
#     grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
#
#     with tf.GradientTape() as tape:
#         conv_outputs, predictions = grad_model(img_array)
#         class_output = predictions[:, class_index]
#
#     grads = tape.gradient(class_output, conv_outputs)
#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
#
#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
#
#     heatmap = np.maximum(heatmap, 0)
#     heatmap /= np.max(heatmap)
#
#     plt.imshow(heatmap)
#     plt.show()
#
# # 图像路径
# image_path = "path_to_your_image.jpg"  # 替换为你的图像路径
#
# # 加载图像并进行预处理
# img_array = load_and_preprocess_image(image_path)
#
# # 使用ResNet-50进行预测
# predictions = model.predict(img_array)
# decoded_predictions = decode_predictions(predictions, top=5)[0]
# print(decoded_predictions)
#
# # 获取最可能的类别索引
# class_index = np.argmax(predictions[0])
# print("Most likely class:", decoded_predictions[0])
#
# # 使用Grad-CAM可视化图像区域重要性
# layer_name = 'conv5_block3_out'  # 选择ResNet-50中的某个卷积层
# visualize_grad_cam(model, img_array, class_index, layer_name)
