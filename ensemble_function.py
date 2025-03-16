import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import json
from torchvision.models import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from model.ensemble.torchensemble.fusion import FusionClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
from postprocess.mcls_roc import mcls_roc_plot
from postprocess import dict_save
from model.ensemble.torchensemble.utils.logging import set_logger

class cnn_model:
    def __init__(self, model_name="resnet34", num_classes=3, model_weight_path=""):
        if model_name == "resnet34":
            self.model = resnet34(pretrained=False)
        elif model_name == "resnet50":
            self.model = resnet50(pretrained=False)
        elif model_name == "resnet101":
            self.model = resnet101(pretrained=False)
        elif model_name == "resnet18":
            self.model = resnet101(pretrained=False)
        elif model_name == "resnet152":
            self.model = resnet101(pretrained=False)
        elif model_name == "vgg19":
            self.model = vgg19(pretrained=False)
        elif model_name == "vgg16":
            self.model = vgg16(pretrained=False)
        elif model_name == "googlenet":
            self.model = googlenet(pretrained=False)
        elif model_name == "alexNet":
            self.model = AlexNet(num_classes=num_classes)
        elif model_name == "densenet121":
            self.model = densenet121(pretrained=False)
        elif model_name == "efficientnet_b0":
            self.model = efficientnet_b0(pretrained=False)
        elif model_name == "efficientnet_v2_s":
            self.model = efficientnet_v2_s(pretrained=False)
        elif model_name == "inception3":
            self.model = inception_v3(pretrained=False)
        elif model_name == "vit_b_16":
            self.model = vit_b_16(pretrained=False)

        else:
            self.model = resnet34(pretrained=False)

        if model_weight_path:
            self.model.load_state_dict(torch.load(model_weight_path))

class CNN_LSTM_Model(nn.Module):
    def __init__(self, cnn_model, num_classes=3, lstm_hidden_size=128, lstm_layers=1):
        super(CNN_LSTM_Model, self).__init__()
        self.cnn = cnn_model
        self.lstm = nn.LSTM(512, lstm_hidden_size, lstm_layers, batch_first=True)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.cnn(x)
        x = x.view(batch_size, -1, 512)  # 展平成适合LSTM输入的形状
        x, _ = self.lstm(x) # 将调整后的特征输入 LSTM 层
        x = x[:, -1, :]  # 取LSTM最后一个时间步的输出
        x = self.fc(x) # 通过全连接层得到最终的分类结果
        return x

def gen_model_folder(estimator_args):
    ensemble_model_name = ""
    for model_info in estimator_args:
        model_name = model_info["model_name"]
        ensemble_model_name = model_name + "-" + ensemble_model_name
        model_weight_path = model_info["model_weight_path"]
        if model_weight_path == None:
            ensemble_model_name = "NP" + "-" + ensemble_model_name
        else:
            ensemble_model_name = "YP" + "-" + ensemble_model_name

    rq = time.strftime("%Y_%m_%d_%H_%M", time.localtime(time.time()))
    ensemble_model_name = ensemble_model_name + rq
    return ensemble_model_name

def endpoint(estimator_args, image_path, save_folder, epochs=1, batch_size=16, learning_rate=0.000005, weight_decay=5e-4, momentum=0.9):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")


    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    print(image_path)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
    train_num = len(train_dataset)
    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    nw = 0

    ensemble_model_name = gen_model_folder(estimator_args)
    save_folder = os.path.join(save_folder, ensemble_model_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    logger = set_logger(ensemble_model_name, use_tb_logger=True, log_path=save_folder)
    import logging
    logger.setLevel(logging.INFO)
    for model_info in estimator_args:
        logger.info(model_info)

    logger.info("device: {}.".format(device))
    logger.info("learning_rate: {}".format(learning_rate))
    logger.info("batch_size: {}".format(batch_size))
    logger.info("epochs: {}".format(epochs))
    logger.info("weight_decay: {}".format(weight_decay))
    logger.info("image_path: {}".format(image_path))
    logger.info('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"), transform=data_transform["val"])
    val_num = len(validate_dataset)
    test_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=batch_size, shuffle=False, num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))

    torch.manual_seed(0)

    # # Initialize models
    base_models = []
    for model_info in estimator_args:
        if model_info["model_name"] == "lstm":
            cnn = resnet34(pretrained=False)
            cnn.fc = nn.Identity()  # 移除最后一层全连接层
            base_model = CNN_LSTM_Model(cnn, num_classes=model_info["num_classes"])
        else:
            base_model = cnn_model(model_info["model_name"], model_info["num_classes"], model_info["model_weight_path"]).model
        base_model.to(device)
        base_models.append(base_model)
    #
    # criterion = nn.CrossEntropyLoss()
    #
    # def train_model(model, dataloader, optimizer):
    #     model.train()
    #     running_loss = 0.0
    #     for inputs, labels in dataloader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #         running_loss += loss.item() * inputs.size(0)
    #     epoch_loss = running_loss / len(dataloader.dataset)
    #     return epoch_loss
    #
    # def evaluate_model(model, dataloader):
    #     model.eval()
    #     all_outputs = []
    #     all_labels = []
    #     with torch.no_grad():
    #         for inputs, labels in dataloader:
    #             inputs, labels = inputs.to(device), labels.to(device)
    #             outputs = model(inputs)
    #             all_outputs.append(outputs.cpu().numpy())
    #             all_labels.append(labels.cpu().numpy())
    #     return np.concatenate(all_outputs), np.concatenate(all_labels)
    #
    # # Train base models 在基础模型训练完之后，使用这些模型对训练集和验证集进行预测，生成元特征
    # for base_model in base_models:
    #     optimizer = torch.optim.Adam(base_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    #     for epoch in range(epochs):
    #         train_loss = train_model(base_model, train_loader, optimizer)
    #         print(f"Epoch {epoch}/{epochs - 1}, Train Loss: {train_loss:.4f}")
    #
    # # Generate meta-features for stacking
    # train_outputs = []
    # train_labels = []
    # val_outputs = []
    # val_labels = []
    # for base_model in base_models:
    #     train_output, train_label = evaluate_model(base_model, train_loader)
    #     val_output, val_label = evaluate_model(base_model, test_loader)
    #     train_outputs.append(train_output)
    #     val_outputs.append(val_output)
    #     if len(train_labels) == 0:
    #         train_labels = train_label
    #         val_labels = val_label
    #
    # train_outputs = np.hstack(train_outputs)
    # val_outputs = np.hstack(val_outputs)
    #
    # # Train meta-classifier 使用Logistic Regression作为元分类器，并在验证集上评估其性能
    # # meta_clf = LogisticRegression(max_iter=1000)
    #
    # #MLP
    # # from sklearn.neural_network import MLPClassifier
    # # meta_clf = MLPClassifier()
    #
    # # from sklearn.svm import SVC
    # # # 创建SVM分类器对象
    # # meta_clf = SVC()
    #
    # from sklearn.ensemble import RandomForestClassifier
    # # 创建随机森林分类器对象
    # meta_clf = RandomForestClassifier()
    #
    # # from sklearn.ensemble import GradientBoostingClassifier
    # # # 创建梯度提升树分类器对象
    # # meta_clf = GradientBoostingClassifier()
    #
    # meta_clf.fit(train_outputs, train_labels)
    #
    # # Evaluate meta-classifier
    # val_predictions = meta_clf.predict(val_outputs)
    # print(confusion_matrix(val_labels, val_predictions))
    # print("accuracy_score: {}".format(accuracy_score(val_labels, val_predictions)))
    # print("classification_report")
    # print(classification_report(val_labels, val_predictions))
    #
    # # Save results
    # output_filename = "results_train.npy"
    # output_path = os.path.join(save_folder, output_filename)
    # dict_save.save_dict_by_numpy(output_path, {"outputs": train_outputs, "labels": train_labels})
    #
    # output_filename = "results_eval.npy"
    # output_path = os.path.join(save_folder, output_filename)
    # dict_save.save_dict_by_numpy(output_path, {"outputs": val_outputs, "labels": val_labels, "predictions": val_predictions})
    #
    # logger.info("computation complete!")
    # return 0
    # Load data
    train_transformer = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )

    test_transformer = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )



    n_estimators = len(estimator_args)
    model = FusionClassifier(

        estimator=cnn_model, n_estimators=n_estimators, cuda=True,estimator_args=estimator_args

    )

    model.set_optimizer("Adam", lr=learning_rate, weight_decay=weight_decay)
    # Training
    tic = time.time()
    results_train,results_eval,best_epoch = model.fit(train_loader, epochs=epochs,test_loader=test_loader,save_dir=save_folder)
    #voting

    index = best_epoch

    print(confusion_matrix(results_eval[index]["target"], results_eval[index]["predicted"]))
    logger.info(confusion_matrix(results_eval[index]["target"], results_eval[index]["predicted"]))

    accuracy_score_val = accuracy_score(results_eval[index]["target"], results_eval[index]["predicted"])
    print("accuracy_score: {}".format(accuracy_score_val))
    logger.info("accuracy_score: {}".format(accuracy_score_val))

    print("classification_report")
    print(classification_report(results_eval[index]["target"], results_eval[index]["predicted"]))
    logger.info(classification_report(results_eval[index]["target"], results_eval[index]["predicted"]))

    toc = time.time()
    training_time = toc - tic
    logger.info("training_time : {}".format(training_time))
    # Evaluating
    # tic = time.time()
    # testing_acc = model.evaluate(test_loader)
    #
    # print("testing_acc:")
    # print(testing_acc)
    # toc = time.time()
    # evaluating_time = toc - tic
    # records = []
    # records.append(
    #     ("FusionClassifier", training_time, evaluating_time, testing_acc)
    # )



    res_labels = np.array(results_eval[index]["predicted"])
    y_score = np.array(results_eval[index]["probs"])
    y_test = np.array(results_eval[index]["target"])

    # folder = "D:\\results\\ensemble_eye_detection\\ensemble\\data\\set_4\\no_pretrained"
    file_name = "roc.jpg"
    fig_path = os.path.join(save_folder,file_name)
    mcls_roc_plot(res_labels, y_score, y_test,cla_dict,fig_path)

    # results_train, results_eval

    output_filename = "results_train.npy"
    output_path = os.path.join(save_folder,output_filename)
    dict_save.save_dict_by_numpy(output_path, results_train)

    output_filename = "results_eval.npy"
    output_path = os.path.join(save_folder,output_filename)
    # 保存
    dict_save.save_dict_by_numpy(output_path, results_eval)
    logger.info("computation complete!")
    result = 0
    return result

