import numpy as np
from sklearn.linear_model import LinearRegression
import json, os
from skimage.io import imread

class LaneEval(object):
    lr = LinearRegression()
    pixel_thresh = 20
    pt_thresh = 0.85

    @staticmethod
    def get_angle(xs, y_samples):
        xs, ys = xs[xs >= 0], y_samples[xs >= 0]
        if len(xs) > 1:
            LaneEval.lr.fit(ys[:, None], xs)
            k = LaneEval.lr.coef_[0]
            theta = np.arctan(k)
        else:
            theta = 0
        return theta

    @staticmethod
    def line_accuracy(pred, gt, thresh):
        pred = np.array([p if p >= 0 else -100 for p in pred])
        gt = np.array([g if g >= 0 else -100 for g in gt])
        return np.sum(np.where(np.abs(pred - gt) < thresh, 1., 0.)) / len(gt)

    @staticmethod
    def get_pred_lanes(pfile_name, tfile_name):
        img = imread(pfile_name, as_gray=True) #读取灰度图
        if tfile_name is not None:
            with open(tfile_name) as f:
                img_data = f.readlines()
            img_data = [line.split() for line in img_data]
            img_data = [list(map(float, lane)) for lane in img_data]
            img_data = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2)]
                        for lane in img_data]
            img_data = [lane for lane in img_data if len(lane) >= 2] #删除长度小于2的车道线

            y_pred = [[img.shape[0]- point[1] for point in line] for line in img_data]
            x_pred = [[point[0] for point in line] for line in img_data]
            param = [np.polyfit(y_pred[k], x_pred[k], 2).tolist() for k in range(len(x_pred))] #2次拟合参数
        else:
            # According to the input image, the corresponding curve of each lane line is obtained
            img_gray = np.unique(img[np.where(img > 0)])
            y_pred = [(img.shape[0] - np.where(img == i)[0] - 1).tolist() for i in img_gray]
            x_pred = [(np.where(img == j)[1] + 1).tolist() for j in img_gray]
            param = [np.polyfit(y_pred[k], x_pred[k], 2).tolist() for k in range(len(x_pred))] #2次拟合参数
        return param, img.shape[0]


    @staticmethod
    def get_gt_lanes(gt_dir, filename, height):
        gt_json = json.load(open(os.path.join(gt_dir, filename))).get('annotations')['lane']
        img_height = height
        lanex_points = []
        laney_points = []
        for i in gt_json:
            for key, value in i.items():
                if key == 'points' and value != []:
                    lanex = []
                    laney = []
                    for item in value:
                        lanex.append(item[0])
                        laney.append(img_height - item[1])
                    lanex_points.append(lanex)
                    laney_points.append(laney)
        return lanex_points,laney_points

    @staticmethod
    def calculate_results(param, gtx, gty): #一张图的内容
        angles = [LaneEval.get_angle(np.array(gtx[i]), np.array(gty[i])) for i in range(len(gty))] #why?
        threshs = [LaneEval.pixel_thresh / np.cos(angle) for angle in angles]
        
        line_accs = []
        fp, fn = 0., 0.
        matched = 0.

        for index, (x_gts, thresh) in enumerate(zip(gtx, threshs)):
            accs = []
            for x_preds in param: #通过拟合参数得到x
                # x_pred = x_preds[0] * np.array(gty[index]) + x_preds[1]
                x_pred = (x_preds[0] * np.array(gty[index]) * np.array(gty[index]) + x_preds[1] * np.array(gty[index]) + x_preds[2]).tolist()
                # x_pred = x_preds[0]*pow(np.array(gty[index]),3) + x_preds[1]*pow(np.array(gty[index]),2) + \
                #          x_preds[2]*pow(np.array(gty[index]),1) + x_preds[3]*pow(np.array(gty[index]),0)
                accs.append(LaneEval.line_accuracy(np.array(x_pred), np.array(x_gts), thresh))
            max_acc = np.max(accs) if len(accs) > 0 else 0. #选取最大的acc 这也算一种匹配?
            line_accs.append(max_acc)
            if max_acc < LaneEval.pt_thresh:
                fn += 1
            else:
                matched += 1
        fp = len(param) - matched
        if len(gtx) > 8 and fn > 0:
            fn -= 1
        s = sum(line_accs)
        if len(gtx) > 8:
            s -= min(line_accs)
        return s / max(min(8.0, len(gtx)), 1.), fp / len(param) if len(param) > 0 else 0., fn / max(min(len(gtx), 8.), 1.)


    @staticmethod
    def calculate_return(pre_dir_name, json_dir_name):
        Preditction = pre_dir_name
        Json = json_dir_name
        pred_txt_path = './txt/pred_txt'
        num, accuracy, fp, fn = 0., 0., 0., 0.
        list_preditction = os.listdir(Preditction)
        list_preditction.sort()
        for filename in list_preditction:
            pred_files = os.listdir(os.path.join(Preditction, filename))
            json_files = os.listdir(os.path.join(Json, filename))
            txt_files = os.listdir(os.path.join(pred_txt_path, filename))
            pred_files.sort()
            json_files.sort()
            txt_files.sort()

            for pfile, jfile, tfile in zip(pred_files, json_files, txt_files): #循环每张图
                pfile_name = os.path.join(Preditction, filename, pfile)
                tfile_name = os.path.join(pred_txt_path, filename, tfile)
                param, height = LaneEval.get_pred_lanes(pfile_name, tfile_name)
                # print('pred_txt_name:', tfile_name)
                # print('pred_image_name:', pfile_name)
                # print('json_file_name:', os.path.join(Json, filename, jfile))
                lanex_points, laney_points = LaneEval.get_gt_lanes(os.path.join(Json, filename), jfile, height)

                try:
                    a, p, n = LaneEval.calculate_results(param, lanex_points, laney_points)
                except BaseException as e:
                    raise Exception('Format of lanes error.')
                accuracy += a
                fp += p
                fn += n
                num += 1

        accuracy = accuracy / num
        fp = fp / num
        fn = fn / num
        F1 = 2*(1-fp)*(1-fn) / (1-fp + 1-fn)
        return accuracy, F1, fp, fn


if __name__ == '__main__':
    # pre_dir_name is the path to your output dir
    pre_dir_name = '/home/chengzy/MMA-TR-Net/output'
    # json_dir_name is the path of Data_ROOT/Json
    json_dir_name = '/home/chengzy/MMA-TR-Net/dataset/VIL100/JsonV4'
    accuracy, F1, fp, fn = LaneEval.calculate_return(pre_dir_name, json_dir_name)
    print({'accuracy': accuracy, 'F1': F1, 'fp': fp, 'fn': fn})






