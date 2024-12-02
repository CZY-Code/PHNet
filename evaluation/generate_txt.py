import os
import cv2
import numpy as np
from skimage.io import imread, imshow
import pickle
from generate_lane import sample_lane

COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]
# ROOT = '../dataset'
def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def generate_pred(pred_path, out_path, json_path):
    #从分割图像生成txt
    pred_dirs = os.listdir(pred_path) #['2_Road036_Trim003_frames','...']
    for pred_dir in pred_dirs: # 2_Road036_Trim003_frames
        print('generate sequence: ',pred_dir)
        pred_dir_path = os.path.join(pred_path, pred_dir)
        all_pred_files = os.listdir(pred_dir_path)
        all_pred_files.sort()
        all_json_files = os.listdir(os.path.join(json_path, pred_dir))
        all_json_files.sort()

        # out_pred_path
        if not os.path.exists(os.path.join(out_path, pred_dir)):
            os.makedirs(os.path.join(out_path, pred_dir))

        for index, img in enumerate(all_pred_files):
            img = imread(os.path.join(pred_dir_path, img), as_gray=True)
            img_gray = np.unique(img[np.where(img > 0)])
            y_pred = [(img.shape[0] - np.where(img == i)[0] - 1).tolist() for i in img_gray]
            x_pred = [(np.where(img == j)[1] + 1).tolist() for j in img_gray]
            param = [np.polyfit(y_pred[k], x_pred[k], 2).tolist() for k in range(len(x_pred))]

            out_file_txt_name = all_json_files[index].replace('jpg.json', 'lines.txt')
            with open(os.path.join(out_path, pred_dir, out_file_txt_name), "w") as fp: #a+:从开头写 w:覆盖
                for i, y in enumerate(y_pred):
                    txty = list(range(min(y), max(y), 3))
                    txtx = (param[i][0] * np.array(txty) * np.array(txty) + param[i][1] * np.array(txty) + param[i][2]).tolist()
                    if len(txtx) >= 2:
                        for index in range(len(txtx)):
                            if txtx[index] >= 0 and txtx[index] <= img.shape[1]:
                                fp.write('%d %d ' % (txtx[index], img.shape[0] - txty[index]))
                        fp.write('\n')

#生成txt标签
def pickle2txt():
    view = False
    srcVideoPath = '/home/chengzy/MMA-TR-Net/dataset/OpenLane/images/validation'
    labelPicklePath = '/home/chengzy/MMA-TR-Net/dataset/OpenLane/OpenLane-V/label/validation'
    targetTxtPath = '/home/chengzy/MMA-TR-Net/evaluation/txt4OL/anno_txt'
    labelDirList = os.listdir(labelPicklePath)
    for videoName in labelDirList:
        labelVidPath = os.path.join(labelPicklePath, videoName)
        txtVideoPath = os.path.join(targetTxtPath, videoName)
        if not os.path.exists(txtVideoPath):
            os.mkdir(txtVideoPath)    
        labelImgList = os.listdir(labelVidPath)
        labelImgList.sort()
        for imgName in labelImgList:
            srcImgPath = os.path.join(srcVideoPath, videoName, imgName[:-7]+'.jpg')
            img = cv2.imread(srcImgPath)
            img_size = img.shape[:2]
            #----load label--------
            lanes_anno = load_pickle(os.path.join(labelVidPath, imgName)) #{'lanes':[4*arrary[N,2]]}
            out_file_txt_name = os.path.join(txtVideoPath, imgName[:-7]+'.lines.txt')
            print(f'generating: {out_file_txt_name}')
            with open(out_file_txt_name, "w") as fp:
                for lane in lanes_anno['lanes']: #循环n条线
                    lane = sample_lane(lane, img_size)
                    for tx, ty in lane: #反转与否对指标没有影响
                        # fp.write('%.1f %.1f ' % (tx, ty))
                        fp.write('%.1f %.1f ' % (tx/2, ty/2)) #XXX 缩短为原来的一半
                    fp.write('\n')

                    if view:
                        xys = []
                        for x, y in lane:
                            if x <= 0 or y <= 0:
                                continue
                            x, y = int(x), int(y)
                            xys.append((x, y))
                            for i in range(1, len(xys)):
                                cv2.line(img, xys[i - 1], xys[i], COLORS[1], thickness=4)
            if view:
                cv2.imshow('img', img)
                key = cv2.waitKey(0)
                if key == 27:
                    exit(0)
    if view:
        cv2.destroyWindow('img')


def pickle2txtPreVid(videoName):
    view = False
    srcVideoPath = '/home/chengzy/MMA-TR-Net/dataset/OpenLane/images/validation'
    targetTxtPath = '/home/chengzy/MMA-TR-Net/evaluation/txt4OL/anno_txt'
    labelPicklePath = '/home/chengzy/MMA-TR-Net/dataset/OpenLane/OpenLane-V/label/validation'
    
    labelVidPath = os.path.join(labelPicklePath, videoName)
    txtVideoPath = os.path.join(targetTxtPath, videoName)
    if not os.path.exists(txtVideoPath):
        os.mkdir(txtVideoPath)    
    labelImgList = os.listdir(labelVidPath)
    labelImgList.sort()
    for imgName in labelImgList:
        srcImgPath = os.path.join(srcVideoPath, videoName, imgName[:-7]+'.jpg')
        img = cv2.imread(srcImgPath)
        img_size = img.shape[:2]
        #----load label--------
        lanes_anno = load_pickle(os.path.join(labelVidPath, imgName)) #{'lanes':[4*arrary[N,2]]}
        out_file_txt_name = os.path.join(txtVideoPath, imgName[:-7]+'.lines.txt')
        print(f'generating: {out_file_txt_name}')
        with open(out_file_txt_name, "w") as fp:
            for lane in lanes_anno['lanes']: #循环n条线
                lane = sample_lane(lane, img_size)
                if len(lane) <= 2:
                    continue
                for tx, ty in lane: #反转与否对指标没有影响 #XXX 缩短为原来的一半
                    tx = tx / (1920 - 1) * (1920 // 2 - 1)
                    ty = ty / (1280 - 1) * (1280 // 2 - 1)
                    fp.write('%.1f %.1f ' % (tx, ty)) 
                fp.write('\n')

                if view:
                    xys = []
                    for x, y in lane:
                        if x <= 0 or y <= 0:
                            continue
                        x, y = int(x), int(y)
                        xys.append((x, y))
                        for i in range(1, len(xys)):
                            cv2.line(img, xys[i - 1], xys[i], COLORS[1], thickness=4)
        if view:
            cv2.imshow('img', img)
            key = cv2.waitKey(0)
            if key == 27:
                exit(0)
    if view:
        cv2.destroyWindow('img')

if __name__ == '__main__':
    # change your pred_path at ${root}/${output}/${valset}
    # pre_dir_name = '/home/chengzy/MMA-TR-Net/output/VIL100/60_lr0.001deay1e-6_sgd' # {pred_dir_name} is the path of your output results
    # json_path = '../dataset/VIL100/Json' # {json_path} is the path of Data_ROOT/Json_ori
    # pred_txt_path = './txt/pred_txt'
    # if not os.path.exists(pred_txt_path):
    #     os.mkdir(pred_txt_path)
    # generate_pred(pre_dir_name, pred_txt_path, json_path)
    
    # pickle2txt()

    #------多线程生成label---------#
    import concurrent.futures
    labelPicklePath = '/home/chengzy/MMA-TR-Net/dataset/OpenLane/OpenLane-V/label/validation'
    labelDirList = os.listdir(labelPicklePath)
    # 定义需要计算的列表
    input_list = [...]
    # 定义一个线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交任务
        futures = [executor.submit(pickle2txtPreVid, videoName) for videoName in labelDirList]
        # 获取结果
        results = [f.result() for f in futures]