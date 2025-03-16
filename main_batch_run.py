import ensemble_function as conn

if __name__ == "__main__":
    epochs = 20  # 总训练周期数
    batch_size = 8  # 批处理大小
    learning_rate = 0.00005  # 学习率
    weight_decay = 5e-4  # 权重衰减
    momentum = 0.9  # 动量
    # image_path = "/opt/data/private/wjj/data/dataset4"  # 图像数据路径
    # image_path = "/opt/data/private/wjj/data/dataset1"  # 图像数据路径
    # image_path = "/opt/data/private/wjj/data/dataset5"  # 图像数据路径
    # image_path = "/opt/data/private/wjj/data/dataset2"  # 图像数据路径
    # image_path = "/opt/data/private/wjj/data/5折数据_dataset4/1"
    # image_path = "/opt/data/private/wjj/data/5折数据_dataset4/2"
    # image_path = "/opt/data/private/wjj/data/5折数据_dataset4/3"
    image_path = "/opt/data/private/wjj/data/5折数据_dataset4/4"
    # image_path = "/opt/data/private/wjj/data/5折数据_dataset4/5"
    # image_path = "/opt/data/private/wjj/data/chest_xray"  # 图像数据路径
    # image_path = "/opt/data/private/wjj/data/DDTI-ult"  # 图像数据路径
    # image_path = "/opt/data/private/zr/dataset/LC2500_train_test"
    # image_path = "/opt/data/private/wjj/data/JZX"  # 图像数据路径
    save_folder = "/opt/data/private/wjj/reslut"  # 结果保存路径

    print("start next running")
    estimator_args = []
    # estimator_args.append({"model_name": "resnet34", "num_classes": 3,
    #                        "model_weight_path": "/opt/data/private/wjj/pretrain/resnet34-333f7ec4.pth"})
    # # #
    # estimator_args.append({"model_name": "vgg16", "num_classes": 3,
    #                        "model_weight_path": "/opt/data/private/wjj/pretrain/vgg16-397923af.pth"})


    # estimator_args.append({"model_name": "lstm", "num_classes": 3, "model_weight_path": None})  # 添加LSTM模型

    # estimator_args.append({"model_name": "resnet50", "num_classes": 3,
    #                      "model_weight_path": "/opt/data/private/wjj/pretrain/resnet50-19c8e357.pth"})
    #
    # estimator_args.append({"model_name": "vgg19", "num_classes": 3,
    #                     "model_weight_path": "/opt/data/private/wjj/pretrain/vgg19-dcbb9e9d.pth"})

    # estimator_args.append({"model_name": "resnet101", "num_classes": 3,
    #                        "model_weight_path": "/opt/data/private/wjj/pretrain/resnet101-5d3b4d8f.pth"})

    #
    estimator_args.append({"model_name": "efficientnet_v2_s", "num_classes": 3,
                           "model_weight_path": "/opt/data/private/wjj/pretrain/efficientnet_v2_s-dd5fe13b.pth"})


    #
    # estimator_args.append({"model_name": "densenet121", "num_classes": 5,
    #                        "model_weight_path": "/opt/data/private/wjj/pretrain/densenet121_weights.pth"})

    result = conn.endpoint(estimator_args, image_path, save_folder, epochs=epochs, batch_size=batch_size)






