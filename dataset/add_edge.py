import cv2
import torch
import numpy as np
from torchvision import transforms, utils

def add_edge_toImage(image_dir, edge_dir, mask_dir):
    image = cv2.imread(image_dir)
    image = cv2.resize(image, (256, 256))
    edge = cv2.imread(edge_dir)
    edge = cv2.resize(edge, (256, 256))
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    if np.random.rand() < 0.6:
        image = cv2.flip(image, 1)
        edge  = cv2.flip(edge, 1)
        #mask = cv2.flip(mask, 1)
    to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    to_tensor_2 = transforms.ToTensor()

    ori_image = image
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    ori_image = to_tensor_2(ori_image)#torch.from_numpy(image)

    white_coords = np.column_stack(np.where(mask > 0))
    #black_coords = np.column_stack(np.where(mask <= 0)) # *********
    # 遍历图像，根据mask中的白色区域坐标将对应像素改成白色
    for coord in white_coords:
        x, y = coord
        image[x, y] = [255, 255, 255]  # 白色像素值
    # 根据mask中的白色区域坐标，将对应的edge中的图像提取并粘贴到A图像上
    for coord in white_coords:
        x, y = coord
        image[x, y] = edge[x, y]
    #cv2.imwrite('datasets/test_DE_edge.jpg', image)
    """for coord in black_coords:
        x, y = coord
        edge[x, y] = [0, 0, 0] """
    #cv2.imwrite('datasets/test_edge.jpg', edge)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = to_tensor_2(image)#torch.from_numpy(image)
    #edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
    #edge = to_tensor_2(edge)
    mask = to_tensor_2(mask)
    image = torch.cat((image, mask), dim=0) # 为了加入mask，在input_layer函数、psp网络loading encoder checkpoint、 train D的_, x = batch处做了修改，并且在计算loss和保存图像时，将x改为x[:, :3, :, :]、更改了calc_loss函数的参数
    #image = image.permute(2, 0, 1).float()
    #cv2.imwrite('datasets/test_DE_edge.jpg', image)
    #image = to_tensor(image)
    #utils.save_image(image, 'datasets/test_edge.jpg')
    return image, ori_image, mask

#add_edge_toImage('/home/liqing/Li/Li_qing/RGTD/data/FFhQ_1024/00000/00001.png', '/home/liqing/Li/Li_qing/RGTD/data/FFhQ_1024_edge_avg/avg_00/00001.png', '/home/liqing/Li/Li_qing/encoder4editing/datasets/mask_256/0000017.png')
def add_edge_toImage_edit(image_dir, edge_dir, mask_dir):
    image = cv2.imread(image_dir)
    image = cv2.resize(image, (256, 256))
    edge = cv2.imread(edge_dir)
    edge = cv2.resize(edge, (256, 256))
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256))

    to_tensor_2 = transforms.ToTensor()

    ori_image = image
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    ori_image = to_tensor_2(ori_image)#torch.from_numpy(image)

    white_coords = np.column_stack(np.where(mask > 0))
    #black_coords = np.column_stack(np.where(mask <= 0)) # *********
    # 遍历图像，根据mask中的白色区域坐标将对应像素改成白色
    for coord in white_coords:
        x, y = coord
        image[x, y] = [255, 255, 255]  # 白色像素值
    # 根据mask中的白色区域坐标，将对应的edge中的图像提取并粘贴到A图像上
    for coord in white_coords:
        x, y = coord
        image[x, y] = edge[x, y]
    #cv2.imwrite('datasets/test_DE_edge.jpg', image)
    """for coord in black_coords:
        x, y = coord
        edge[x, y] = [0, 0, 0] """
    #cv2.imwrite('datasets/test_edge.jpg', edge)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = to_tensor_2(image)#torch.from_numpy(image)
    #edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
    #edge = to_tensor_2(edge)
    mask = to_tensor_2(mask)
    image = torch.cat((image, mask), dim=0) # 为了加入mask，在input_layer函数、psp网络loading encoder checkpoint、 train D的_, x = batch处做了修改，并且在计算loss和保存图像时，将x改为x[:, :3, :, :]、更改了calc_loss函数的参数
    #image = image.permute(2, 0, 1).float()
    #cv2.imwrite('datasets/test_DE_edge.jpg', image)
    #image = to_tensor(image)
    #utils.save_image(image, 'datasets/test_edge.jpg')
    return image, ori_image, mask

def add_edge_toImage_edit_without(image_dir, edge_dir, mask_dir):
    image = cv2.imread(image_dir)
    image = cv2.resize(image, (256, 256))
    mask = cv2.imread(mask_dir, cv2.IMREAD_GRAYSCALE)

    to_tensor_2 = transforms.ToTensor()

    ori_image = image
    ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    ori_image = to_tensor_2(ori_image)#torch.from_numpy(image)

    white_coords = np.column_stack(np.where(mask > 0))
    #black_coords = np.column_stack(np.where(mask <= 0)) # *********
    # 遍历图像，根据mask中的白色区域坐标将对应像素改成白色
    for coord in white_coords:
        x, y = coord
        image[x, y] = [255, 255, 255]  # 白色像素值
    # 根据mask中的白色区域坐标，将对应的edge中的图像提取并粘贴到A图像上
    #cv2.imwrite('datasets/test_DE_edge.jpg', image)
    """for coord in black_coords:
        x, y = coord
        edge[x, y] = [0, 0, 0] """
    #cv2.imwrite('datasets/test_edge.jpg', edge)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = to_tensor_2(image)#torch.from_numpy(image)
    #edge = cv2.cvtColor(edge, cv2.COLOR_BGR2RGB)
    #edge = to_tensor_2(edge)
    mask = to_tensor_2(mask)
    #image = torch.cat((image, mask), dim=0) # 为了加入mask，在input_layer函数、psp网络loading encoder checkpoint、 train D的_, x = batch处做了修改，并且在计算loss和保存图像时，将x改为x[:, :3, :, :]、更改了calc_loss函数的参数
    #image = image.permute(2, 0, 1).float()
    #cv2.imwrite('datasets/test_DE_edge.jpg', image)
    #image = to_tensor(image)
    #utils.save_image(image, 'datasets/test_edge.jpg')
    return image, ori_image, mask