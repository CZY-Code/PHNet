import cv2
import numpy as np
import os
#------------------Grad-cam----------------------------------
def cam_show_img(img, feature_map, grads, out_dir):
    H, W, _ = img.shape
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)		# 4
    for i in range(8): #feature_map.shape[0]
        fmap = feature_map[i, :, :]
        fmap = fmap / np.absolute(fmap).max()
        # fmap = cv2.resize(fmap, (W, H))
        fmap = cv2.applyColorMap(np.uint8(255 * fmap), cv2.COLORMAP_RAINBOW)
        cv2.imshow('fmap_{}'.format(i),  fmap)

    grads = grads.reshape([grads.shape[0], -1])#[9, 320*640]
    weights = np.mean(grads, axis=1)		   #[9]
    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]		    
    # cam = np.maximum(cam, 0)
    cam = cam / np.absolute(cam).max()
    cam = cv2.resize(cam, (W, H))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    cam_img = 0.3 * heatmap + 0.7 * img

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    cv2.imshow('cam_img', cam_img)
    cv2.imwrite(path_cam_img, heatmap)
    cv2.waitKey(0)

def grad_cam(model, train_loader):
    fmap_block = list()
    grad_block = list()
    output_dir = './cam'
    def farward_hook(module, input, output):
        return fmap_block.append(output[0])
    def backward_hook(module, grad_in, grad_out):
        return grad_block.append(grad_out[0].detach())

    model.net.module.neck.fpn_convs[0].register_forward_hook(farward_hook)
    model.net.module.neck.fpn_convs[0].register_forward_hook(backward_hook)
    model.net.train()
    for i, data in enumerate(train_loader):
        if model.recorder.step >= model.cfg.total_iter:
            break
        model.recorder.step += 1
        img = data['img'][0].permute(1,2,0)
        data = model.to_cuda(data)
        model.net.zero_grad()
        output = model.net(data)
        loss = output['loss'].sum()
        loss.backward()

        img = img.cpu().data.numpy()
        grads_val = grad_block[-1].cpu().data.numpy().squeeze()
        fmap = fmap_block[-1].cpu().data.numpy().squeeze()

        # 保存cam图片
        cam_show_img(img, fmap, grads_val, output_dir)
        if(len(fmap_block)>10):
            fmap_block.pop(0)
            grad_block.pop(0)
