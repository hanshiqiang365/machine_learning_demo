#author:hanshiqiang365

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["SimHei"]

def ks_plot(predictions, labels, cut_point=100):
    good_len = len([x for x  in labels if x == 0])  # 所有好雇主数量
    bad_len = len([x for x in labels if x == 1])  # 所有坏雇主数量
    predictions_labels = list(zip(predictions, labels))
    good_point = []
    bad_point = []
    diff_point = []  # 记录每个阈值点下的KS值

    x_axis_range = np.linspace(0, 1, cut_point)
    for i in x_axis_range:
        hit_data = [x[1] for x in predictions_labels if x[0] <= i]  # 选取当前阈值下的数据
        good_hit = len([x for x in hit_data if x == 0])  # 预测好雇主数量
        bad_hit = len([x for x in hit_data if x == 1])  # 预测坏雇主数量
        good_rate = good_hit / good_len  # 预测好雇主占比总好客户数
        bad_rate = bad_hit / bad_len  # 预测坏雇主占比总坏客户数
        diff = good_rate - bad_rate  # KS值
        good_point.append(good_rate)
        bad_point.append(bad_rate)
        diff_point.append(diff)

    ks_value = max(diff_point)  # 获得最大KS值为KS值
    ks_x_axis = diff_point.index(ks_value)  # KS值下的阈值点索引
    ks_good_point, ks_bad_point = good_point[ks_x_axis], bad_point[ks_x_axis]  # 阈值下好坏雇主在组内的占比
    threshold = x_axis_range[ks_x_axis]  # 阈值

    plt.plot(x_axis_range, good_point, color="green", label="好雇主比率")
    plt.plot(x_axis_range, bad_point, color="red", label="坏雇主比例")
    plt.plot(x_axis_range, diff_point, color="darkorange", alpha=0.5)
    plt.plot([threshold, threshold], [0, 1], linestyle="--", color="black", alpha=0.3, linewidth=2)
    
    plt.scatter([threshold], [ks_good_point], color="white", edgecolors="green", s=15)
    plt.scatter([threshold], [ks_bad_point], color="white", edgecolors="red", s=15)
    plt.scatter([threshold], [ks_value], color="white", edgecolors="darkorange", s=15)
    plt.title("KS={:.3f} threshold={:.3f}".format(ks_value, threshold))
    
    plt.text(threshold + 0.02, ks_good_point + 0.05, round(ks_good_point, 2))
    plt.text(threshold + 0.02, ks_bad_point + 0.05, round(ks_bad_point, 2))
    plt.text(threshold + 0.02, ks_value + 0.05, round(ks_value, 2))
    
    plt.legend(loc=4)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    # 读取预测数据和真实标签
    labels = []
    predictions = []
    with open("test_predict_example.txt", "r", encoding="utf8") as f:
        for line in f.readlines():
            labels.append(float(line.strip().split()[0]))
            predictions.append(float(line.strip().split()[1]))

    ks_plot(predictions, labels)
