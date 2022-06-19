
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT)) 
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) 

from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync


@torch.no_grad()
def run(source):
    weights= 'last.pt'
    data='data/custom_data.yaml'
    imgsz=(640, 640)
    conf_thres=0.25
    iou_thres=0.45
    max_det=1000
    device=''
    view_img=False
    save_txt=False
    save_conf=False
    save_crop=False
    nosave=False
    classes=None
    agnostic_nms=False
    augment=False
    visualize=False  
    update=False 
    project=ROOT / 'runs/detect'
    name=''  
    exist_ok=False 
    line_thickness=3
    hide_labels=False
    hide_conf=False
    half=False
    dnn=False
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source) 

   
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok) 
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True) 

   
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  

    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  
    vid_path, vid_writer = [None] * bs, [None] * bs

    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model.fp16 else im.float() 
        im /= 255 
        if len(im.shape) == 3:
            im = im[None] 
        t2 = time_sync()
        dt[0] += t2 - t1


        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2


        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

      
        for i, det in enumerate(pred):  
            seen += 1
            if webcam:  
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
                
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  
            save_path = str(save_dir / p.name)  
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  
            
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  
            imc = im0.copy() if save_crop else im0 
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum() 
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  

                
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist() 
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh) 
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img: 
                        c = int(cls)
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

          
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1) 

           
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite('bodylang.jpg', im0)
                else:  
                    if vid_path[i] != save_path:  
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  
                        if vid_cap:  
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

       
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
     
        
        return str(names[c])
       

   
  
