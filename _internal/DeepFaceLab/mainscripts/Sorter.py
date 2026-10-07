import math
import multiprocessing
import operator
import os
import sys
import tempfile
from functools import cmp_to_key
from pathlib import Path

import cv2
import numpy as np
from numpy import linalg as npla

from core import imagelib, mathlib, pathex
from core.cv2ex import *
from core.imagelib import estimate_sharpness
from core.interact import interact as io
from core.joblib import Subprocessor
from core.faceset_transaction import execute_plan
from core.faceset_transaction import sha256 as file_sha256
from core.imagelib.face_quality import DEFAULT_METRIC, METRICS, face_detail_signals, metadata_coverage, select_quality_coverage
from core.leras import nn
from DFLIMG import *
from facelib import LandmarksProcessor


class BlurEstimatorSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        def on_initialize(self, client_dict):
            self.estimate_motion_blur = client_dict['estimate_motion_blur']

        #override
        def process_data(self, data):
            filepath = Path( data[0] )
            dflimg = DFLIMG.load (filepath)

            if dflimg is None or not dflimg.has_data():
                self.log_err (f"{filepath.name} 不是 DFL 图像文件")
                return [ str(filepath), 0 ]
            else:
                image = cv2_imread( str(filepath) )

                face_mask = LandmarksProcessor.get_image_hull_mask (image.shape, dflimg.get_landmarks())
                image = (image*face_mask).astype(np.uint8)


                if self.estimate_motion_blur:
                    value = cv2.Laplacian(image, cv2.CV_64F, ksize=11).var()
                else:
                    value = estimate_sharpness(image)

                return [ str(filepath), value ]


        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, input_data, estimate_motion_blur=False ):
        self.input_data = input_data
        self.estimate_motion_blur = estimate_motion_blur
        self.img_list = []
        self.trash_img_list = []
        super().__init__('BlurEstimator', BlurEstimatorSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("", len (self.input_data))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close ()

    #override
    def process_info_generator(self):
        cpu_count = multiprocessing.cpu_count()
        io.log_info(f'正在使用 {cpu_count} 个 CPU 线程/进程运行')

        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {'estimate_motion_blur':self.estimate_motion_blur}

    #override
    def get_data(self, host_dict):
        if len (self.input_data) > 0:
            return self.input_data.pop(0)

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.input_data.insert(0, data)

    #override
    def on_result (self, host_dict, data, result):
        if result[1] == 0:
            self.trash_img_list.append ( result )
        else:
            self.img_list.append ( result )

        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.img_list, self.trash_img_list


def sort_by_blur(input_path):
    io.log_info ("正在按清晰度（blur）排序...")

    img_list = [ (filename,[]) for filename in pathex.get_image_paths(input_path) ]
    img_list, trash_img_list = BlurEstimatorSubprocessor (img_list).run()

    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

def sort_by_motion_blur(input_path):
    io.log_info ("正在按运动模糊（motion blur）排序...")

    img_list = [ (filename,[]) for filename in pathex.get_image_paths(input_path) ]
    img_list, trash_img_list = BlurEstimatorSubprocessor (img_list, estimate_motion_blur=True).run()

    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

def sort_by_face_yaw(input_path):
    io.log_info ("正在按人脸偏航（yaw）排序...")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "加载中"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} 不是 DFL 图像文件")
            trash_img_list.append ( [str(filepath)] )
            continue

        pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks(), size=dflimg.get_shape()[1] )

        img_list.append( [str(filepath), yaw ] )

    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

def sort_by_face_pitch(input_path):
    io.log_info ("正在按人脸俯仰（pitch）排序...")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "加载中"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} 不是 DFL 图像文件")
            trash_img_list.append ( [str(filepath)] )
            continue

        pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks(), size=dflimg.get_shape()[1] )

        img_list.append( [str(filepath), pitch ] )

    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list

def sort_by_face_source_rect_size(input_path):
    io.log_info ("正在按源图人脸框大小排序...")
    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "加载中"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} 不是 DFL 图像文件")
            trash_img_list.append ( [str(filepath)] )
            continue

        source_rect = dflimg.get_source_rect()
        rect_area = mathlib.polygon_area(np.array(source_rect[[0,2,2,0]]).astype(np.float32), np.array(source_rect[[1,1,3,3]]).astype(np.float32))

        img_list.append( [str(filepath), rect_area ] )

    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)

    return img_list, trash_img_list



class HistSsimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            img_list = []
            for x in data:
                img = cv2_imread(x)
                img_list.append ([x, cv2.calcHist([img], [0], None, [256], [0, 256]),
                                     cv2.calcHist([img], [1], None, [256], [0, 256]),
                                     cv2.calcHist([img], [2], None, [256], [0, 256])
                                 ])

            img_list_len = len(img_list)
            for i in range(img_list_len-1):
                min_score = float("inf")
                j_min_score = i+1
                for j in range(i+1,len(img_list)):
                    score = cv2.compareHist(img_list[i][1], img_list[j][1], cv2.HISTCMP_BHATTACHARYYA) + \
                            cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA) + \
                            cv2.compareHist(img_list[i][3], img_list[j][3], cv2.HISTCMP_BHATTACHARYYA)
                    if score < min_score:
                        min_score = score
                        j_min_score = j
                img_list[i+1], img_list[j_min_score] = img_list[j_min_score], img_list[i+1]

                self.progress_bar_inc(1)

            return img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, img_list ):
        self.img_list = img_list
        self.img_list_len = len(img_list)

        slice_count = 20000
        sliced_count = self.img_list_len // slice_count

        if sliced_count > 12:
            sliced_count = 11.9
            slice_count = int(self.img_list_len / sliced_count)
            sliced_count = self.img_list_len // slice_count

        self.img_chunks_list = [ self.img_list[i*slice_count : (i+1)*slice_count] for i in range(sliced_count) ] + \
                               [ self.img_list[sliced_count*slice_count:] ]

        self.result = []
        super().__init__('HistSsim', HistSsimSubprocessor.Cli, 0)

    #override
    def process_info_generator(self):
        cpu_count = len(self.img_chunks_list)
        io.log_info(f'正在使用 {cpu_count} 个线程运行')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {'i':i}, {}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("排序", len(self.img_list))
        io.progress_bar_inc(len(self.img_chunks_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.img_chunks_list) > 0:
            return self.img_chunks_list.pop(0)
        return None

    #override
    def on_data_return (self, host_dict, data):
        raise Exception("处理数据失败。请减少图片数量后重试。")

    #override
    def on_result (self, host_dict, data, result):
        self.result += result
        return 0

    #override
    def get_result(self):
        return self.result

def sort_by_hist(input_path):
    io.log_info ("正在按直方图相似度排序...")
    img_list = HistSsimSubprocessor(pathex.get_image_paths(input_path)).run()
    return img_list, []

class HistDissimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.img_list = client_dict['img_list']
            self.img_list_len = len(self.img_list)

        #override
        def process_data(self, data):
            i = data[0]
            score_total = 0
            for j in range( 0, self.img_list_len):
                if i == j:
                    continue
                score_total += cv2.compareHist(self.img_list[i][1], self.img_list[j][1], cv2.HISTCMP_BHATTACHARYYA)

            return score_total

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return self.img_list[data[0]][0]

    #override
    def __init__(self, img_list ):
        self.img_list = img_list
        self.img_list_range = [i for i in range(0, len(img_list) )]
        self.result = []
        super().__init__('HistDissim', HistDissimSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("排序", len (self.img_list) )

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), 8)
        io.log_info(f'正在使用 {cpu_count} 个 CPU 线程/进程运行')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {'img_list' : self.img_list}

    #override
    def get_data(self, host_dict):
        if len (self.img_list_range) > 0:
            return [self.img_list_range.pop(0)]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.img_list_range.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        self.img_list[data[0]][2] = result
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.img_list

def sort_by_hist_dissim(input_path):
    io.log_info ("正在按直方图差异度排序...")

    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "加载中"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        image = cv2_imread(str(filepath))

        if dflimg is not None and dflimg.has_data():
            face_mask = LandmarksProcessor.get_image_hull_mask (image.shape, dflimg.get_landmarks())
            image = (image*face_mask).astype(np.uint8)

        img_list.append ([str(filepath), cv2.calcHist([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0, 256]), 0 ])

    img_list = HistDissimSubprocessor(img_list).run()

    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(2), reverse=True)

    return img_list, trash_img_list

def sort_by_brightness(input_path):
    io.log_info ("正在按亮度排序...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2_imread(x), cv2.COLOR_BGR2HSV)[...,2].flatten()  )] for x in io.progress_bar_generator( pathex.get_image_paths(input_path), "加载中") ]
    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    return img_list, []

def sort_by_hue(input_path):
    io.log_info ("正在按色相排序...")
    img_list = [ [x, np.mean ( cv2.cvtColor(cv2_imread(x), cv2.COLOR_BGR2HSV)[...,0].flatten()  )] for x in io.progress_bar_generator( pathex.get_image_paths(input_path), "加载中") ]
    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=True)
    return img_list, []

def sort_by_black(input_path):
    io.log_info ("正在按黑色像素数量排序...")

    img_list = []
    for x in io.progress_bar_generator( pathex.get_image_paths(input_path), "加载中"):
        img = cv2_imread(x)
        img_list.append ([x, img[(img == 0)].size ])

    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1), reverse=False)

    return img_list, []

def sort_by_origname(input_path):
    io.log_info ("正在按原始文件名排序...")

    img_list = []
    trash_img_list = []
    for filepath in io.progress_bar_generator( pathex.get_image_paths(input_path), "加载中"):
        filepath = Path(filepath)

        dflimg = DFLIMG.load (filepath)

        if dflimg is None or not dflimg.has_data():
            io.log_err (f"{filepath.name} 不是 DFL 图像文件")
            trash_img_list.append( [str(filepath)] )
            continue

        img_list.append( [str(filepath), dflimg.get_source_filename()] )

    io.log_info ("排序中...")
    img_list = sorted(img_list, key=operator.itemgetter(1))
    return img_list, trash_img_list

def sort_by_oneface_in_image(input_path):
    io.log_info ("正在按每张图片的单脸分组排序...")
    image_paths = pathex.get_image_paths(input_path)
    a = np.array ([ ( int(x[0]), int(x[1]) ) \
                      for x in [ Path(filepath).stem.split('_') for filepath in image_paths ] if len(x) == 2
                  ])
    if len(a) > 0:
        idxs = np.ndarray.flatten ( np.argwhere ( a[:,1] != 0 ) )
        idxs = np.unique ( a[idxs][:,0] )
        idxs = np.ndarray.flatten ( np.argwhere ( np.array([ x[0] in idxs for x in a ]) == True ) )
        if len(idxs) > 0:
            io.log_info ("找到 %d 张图片。" % (len(idxs)) )
            img_list = [ (path,) for i,path in enumerate(image_paths) if i not in idxs ]
            trash_img_list = [ (image_paths[x],) for x in idxs ]
            return img_list, trash_img_list

    io.log_info ("未找到结果。你可能需要先恢复原始文件名。")
    return [], []

class FinalLoaderSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def on_initialize(self, client_dict):
            self.faster = client_dict['faster']

        #override
        def process_data(self, data):
            filepath = Path(data[0])

            try:
                dflimg = DFLIMG.load (filepath)

                if dflimg is None or not dflimg.has_data():
                    self.log_err (f"{filepath.name} 不是 DFL 图像文件")
                    return [ 1, [str(filepath)] ]

                bgr = cv2_imread(str(filepath))
                if bgr is None:
                    raise Exception ("无法加载 %s" % (filepath.name) )

                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                if self.faster:
                    source_rect = dflimg.get_source_rect()
                    if source_rect is None:
                        # Missing source boxes do not mean a defective face.
                        # Keep the image as a zero/unknown metric candidate.
                        sharpness = 0.0
                    else:
                        source_rect = np.asarray(source_rect, dtype=np.float32)
                        if source_rect.shape != (4,) or not np.isfinite(source_rect).all():
                            raise ValueError('源图人脸框必须是四个有限坐标')
                        sharpness = max(0.0, float(source_rect[2] - source_rect[0])) * max(0.0, float(source_rect[3] - source_rect[1]))
                else:
                    face_mask = LandmarksProcessor.get_image_hull_mask (gray.shape, dflimg.get_landmarks())
                    sharpness = estimate_sharpness( (gray[...,None]*face_mask).astype(np.uint8) )

                pitch, yaw, roll = LandmarksProcessor.estimate_pitch_yaw_roll ( dflimg.get_landmarks(), size=dflimg.get_shape()[1] )

                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            except Exception as e:
                self.log_err (e)
                return [ 1, [str(filepath)] ]

            return [0, [str(filepath), sharpness, hist, yaw, pitch,
                        {'metric': 'source_box_area' if self.faster else 'foreground_cpbd',
                         'source_box_available': dflimg.get_source_rect() is not None}]]

        #override
        def get_data_name (self, data):
            #return string identificator of your data
            return data[0]

    #override
    def __init__(self, img_list, faster ):
        self.img_list = img_list

        self.faster = faster
        self.result = []
        self.result_trash = []

        super().__init__('FinalLoader', FinalLoaderSubprocessor.Cli, 60)

    #override
    def on_clients_initialized(self):
        io.progress_bar ("加载", len (self.img_list))

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), 8)
        io.log_info(f'正在使用 {cpu_count} 个 CPU 线程/进程运行')

        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {'faster': self.faster}

    #override
    def get_data(self, host_dict):
        if len (self.img_list) > 0:
            return [self.img_list.pop(0)]

        return None

    #override
    def on_data_return (self, host_dict, data):
        self.img_list.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        if result[0] == 0:
            self.result.append (result[1])
        else:
            self.result_trash.append (result[1])
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result, self.result_trash

class FinalHistDissimSubprocessor(Subprocessor):
    class Cli(Subprocessor.Cli):
        #override
        def process_data(self, data):
            idx, pitch_yaw_img_list = data

            for p in range ( len(pitch_yaw_img_list) ):

                img_list = pitch_yaw_img_list[p]
                if img_list is not None:
                    for i in range( len(img_list) ):
                        score_total = 0
                        for j in range( len(img_list) ):
                            if i == j:
                                continue
                            score_total += cv2.compareHist(img_list[i][2], img_list[j][2], cv2.HISTCMP_BHATTACHARYYA)
                        img_list[i][3] = score_total

                    pitch_yaw_img_list[p] = sorted(img_list, key=operator.itemgetter(3), reverse=True)

            return idx, pitch_yaw_img_list

        #override
        def get_data_name (self, data):
            return "Bunch of images"

    #override
    def __init__(self, pitch_yaw_sample_list ):
        self.pitch_yaw_sample_list = pitch_yaw_sample_list
        self.pitch_yaw_sample_list_len = len(pitch_yaw_sample_list)

        self.pitch_yaw_sample_list_idxs = [ i for i in range(self.pitch_yaw_sample_list_len) if self.pitch_yaw_sample_list[i] is not None ]
        self.result = [ None for _ in range(self.pitch_yaw_sample_list_len) ]
        super().__init__('FinalHistDissimSubprocessor', FinalHistDissimSubprocessor.Cli)

    #override
    def process_info_generator(self):
        cpu_count = min(multiprocessing.cpu_count(), 8)
        io.log_info(f'正在使用 {cpu_count} 个 CPU 线程/进程运行')
        for i in range(cpu_count):
            yield 'CPU%d' % (i), {}, {}

    #override
    def on_clients_initialized(self):
        io.progress_bar ("按直方图差异度排序", len(self.pitch_yaw_sample_list_idxs) )

    #override
    def on_clients_finalized(self):
        io.progress_bar_close()

    #override
    def get_data(self, host_dict):
        if len (self.pitch_yaw_sample_list_idxs) > 0:
            idx = self.pitch_yaw_sample_list_idxs.pop(0)

            return idx, self.pitch_yaw_sample_list[idx]
        return None

    #override
    def on_data_return (self, host_dict, data):
        self.pitch_yaw_sample_list_idxs.insert(0, data[0])

    #override
    def on_result (self, host_dict, data, result):
        idx, yaws_sample_list = data
        self.result[idx] = yaws_sample_list
        io.progress_bar_inc(1)

    #override
    def get_result(self):
        return self.result

def sort_best_faster(input_path, target_count=None):
    return sort_best(input_path, faster=True, target_count=target_count)


def select_best_samples(samples, target_count):
    """Exact-count legacy quality/pose coverage selection, without file writes.

    Metric index 1 is explicitly either CPBD sharpness or source-box area.
    It is not an identity score or a perceptual-model accuracy claim. Occupied
    yaw/pitch bins share the budget; unavailable bins never discard that budget.
    """
    if type(target_count) is not int or target_count < 1:
        raise ValueError('目标人脸数量必须是正整数')
    target_count = min(target_count, len(samples))
    if target_count == 0:
        return [], []
    buckets = {}
    for sample in samples:
        if len(sample) < 5 or not np.isfinite([sample[1], sample[3], sample[4]]).all():
            raise ValueError('人脸筛选需要有限的评分、yaw 和 pitch')
        yaw_bin = int(np.clip((float(sample[3]) + 1.2) / 2.4 * 128, 0, 127))
        pitch_bin = int(np.clip((float(sample[4]) + math.pi / 2) / math.pi * 8, 0, 7))
        buckets.setdefault((yaw_bin, pitch_bin), []).append(sample)
    for items in buckets.values():
        items.sort(key=lambda item: (-float(item[1]), str(item[0]).casefold()))
    # When target_count is smaller than the occupied bin count, start from
    # the strongest candidate, then choose the most distant occupied pose.
    # This retains profile/pitch coverage instead of preferentially keeping
    # only frontal faces or accidentally assigning every bin a zero quota.
    remaining = set(buckets)
    first = min(remaining, key=lambda key: (-float(buckets[key][0][1]), key))
    ordered_bins = [first]; remaining.remove(first)
    distances = {key: float('inf') for key in remaining}
    while remaining:
        chosen = ordered_bins[-1]
        for candidate in remaining:
            distances[candidate] = min(distances[candidate],
                                       ((candidate[0] - chosen[0]) / 127) ** 2 + ((candidate[1] - chosen[1]) / 7) ** 2)
        key = min(remaining, key=lambda candidate: (
            -distances[candidate],
            -float(buckets[candidate][0][1]), candidate))
        ordered_bins.append(key); remaining.remove(key)
    selected = []
    while len(selected) < target_count:
        for key in ordered_bins:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                if len(selected) == target_count: break
    rejected = [item for key in ordered_bins for item in buckets[key]]
    return selected, rejected


def sort_best(input_path, faster=False, target_count=None):
    if target_count is None:
        target_count = io.input_int('目标人脸数量？', 2000)
    if type(target_count) is not int or target_count < 1:
        raise ValueError('目标人脸数量必须是正整数')
    io.log_info('正在筛选姿态覆盖样本。')
    io.log_info('筛选指标：源图人脸框面积（缺失记为未知/0）' if faster else '筛选指标：前景 CPBD 清晰度')
    samples, unreadable = FinalLoaderSubprocessor(pathex.get_image_paths(input_path), faster).run()
    selected, rejected = select_best_samples(samples, target_count)
    io.log_info(f'可读样本 {len(samples)}，选中 {len(selected)}，未选中 {len(rejected)}，不可读 {len(unreadable)}；未选中原件保存在独立批次。')
    return selected, rejected + unreadable


def sort_quality_coverage(input_path, target_count=None, *, offset=0, limit=500, quality_metric=None):
    """Score a visible <=500 window; preserve every item outside that window."""
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 500:
        raise ValueError('质量筛选需要 offset>=0 和 1..500 的 limit')
    quality_metric = quality_metric or DEFAULT_METRIC
    if quality_metric not in METRICS:
        raise ValueError('未知质量代理指标')
    paths = pathex.get_image_paths(input_path)
    batch = paths[offset:offset + limit]
    if not batch:
        raise ValueError(f'指定批次为空：offset={offset}，数据集共 {len(paths)} 张')
    if target_count is None:
        target_count = min(2000, len(batch))
    if type(target_count) is not int or target_count < 1:
        raise ValueError('目标人脸数量必须是正整数')
    io.log_info(f'质量代理筛选范围 {offset + 1}–{offset + len(batch)} / {len(paths)}，每轮最多 500；眼口状态仅是既有关键点几何分桶。')
    samples, unreadable = [], []
    for filename in io.progress_bar_generator(batch, '前景细节与覆盖分析'):
        path = Path(filename)
        digest = file_sha256(path)
        evidence = {'sourceSha256': digest, 'metric': quality_metric,
                    'annotationAccuracyAvailable': False, 'identityQualityAvailable': False}
        try:
            dfl = DFLIMG.load(path)
            if dfl is None or not dfl.has_data():
                raise ValueError('不是具有既有关键点的 DFL aligned')
            image = cv2_imread(str(path))
            if image is None or image.shape[0] != image.shape[1]:
                raise ValueError('需要可读的正方形 aligned 图片')
            points = dfl.get_landmarks()
            face_mask = LandmarksProcessor.get_image_hull_mask(image.shape, points)[..., 0]
            signals = face_detail_signals(image, points, foreground=face_mask)
            pose = LandmarksProcessor.estimate_pitch_yaw_roll(points, size=image.shape[1])
            coverage = metadata_coverage(points, pose, canvas_size=image.shape[1])
            rect = dfl.get_source_rect()
            area = None
            if rect is not None:
                rect = np.asarray(rect, dtype=np.float64)
                if rect.shape == (4,) and np.isfinite(rect).all() and rect[2] > rect[0] and rect[3] > rect[1]:
                    area = float((rect[2] - rect[0]) * (rect[3] - rect[1]))
                    if not math.isfinite(area): area = None
            cpbd = float(estimate_sharpness(np.uint8(image * face_mask[..., None])))
            if not math.isfinite(cpbd): raise ValueError('CPBD 基线返回非有限值')
            evidence.update({'source_box_available': area is not None,
                'sourceBoxArea': area, 'legacyForegroundCpbd': cpbd,
                'detail': signals, 'coverage': coverage, 'available': True})
            score = signals['scores'][quality_metric]
            samples.append([str(path), score, None, float(pose[1]), float(pose[0]), evidence])
        except (ValueError, TypeError, AttributeError, cv2.error, IndexError) as error:
            evidence.update({'available': False, 'reason': str(error),
                             'selectionReason': 'unavailable-existing-metadata-or-image'})
            unreadable.append([str(path), 0., None, 0., 0., evidence])
        if file_sha256(path) != digest:
            raise ValueError(f'分析期间源图发生变化：{path.name}')
    selected, rejected = select_quality_coverage(samples, target_count)
    scored = {str(Path(row[0]).absolute()) for row in selected + rejected + unreadable}
    preserved = [path for path in paths if str(Path(path).absolute()) not in scored]
    context = {'method': quality_metric, 'targetCount': target_count,
               'selectedRange': {'start': offset + 1, 'end': offset + len(batch),
                                 'total': len(paths), 'offset': offset, 'limit': limit, 'count': len(batch)},
               'candidateCount': len(batch), 'preservedOutsideCount': len(preserved),
               'policy': 'interpretable-proxies; preserve-pose-and-geometric-eye-mouth-coverage; originals-recoverable',
               'targetPrefix': f'quality-{offset:06d}-'}
    return selected, rejected + unreadable, preserved, context

"""
def sort_by_vggface(input_path):
    io.log_info ("正在使用 VGGFace 模型按人脸相似度排序...")

    model = VGGFace()

    final_img_list = []
    trash_img_list = []

    image_paths = pathex.get_image_paths(input_path)
    img_list = [ (x,) for x in image_paths ]
    img_list_len = len(img_list)
    img_list_range = [*range(img_list_len)]

    feats = [None]*img_list_len
    for i in io.progress_bar_generator(img_list_range, "加载中"):
        img = cv2_imread( img_list[i][0] ).astype(np.float32)
        img = imagelib.normalize_channels (img, 3)
        img = cv2.resize (img, (224,224) )
        img = img[..., ::-1]
        img[..., 0] -= 93.5940
        img[..., 1] -= 104.7624
        img[..., 2] -= 129.1863
        feats[i] = model.predict( img[None,...] )[0]

    tmp = np.zeros( (img_list_len,) )
    float_inf = float("inf")
    for i in io.progress_bar_generator ( range(img_list_len-1), "排序中" ):
        i_feat = feats[i]

        for j in img_list_range:
            tmp[j] = npla.norm(i_feat-feats[j]) if j >= i+1 else float_inf

        idx = np.argmin(tmp)

        img_list[i+1], img_list[idx] = img_list[idx], img_list[i+1]
        feats[i+1], feats[idx] = feats[idx], feats[i+1]

    return img_list, trash_img_list
"""

def sort_by_absdiff(input_path):
    """Greedy exact pixel-distance ordering with bounded memory and no TF."""
    io.log_info("正在按绝对差异（absdiff）排序...")
    image_paths = pathex.get_image_paths(input_path)
    if len(image_paths) < 2:
        return [(filename,) for filename in image_paths], []
    is_sim = io.input_bool("按相似排序？", True, help_message="否则将按不相似排序。")
    first = cv2_imread(image_paths[0])
    if first is None or first.ndim != 3:
        raise ValueError(f"无法读取图片：{image_paths[0]}")
    shape = first.shape
    pixel_count = int(np.prod(shape))
    batch_size = max(1, min(128, (8 * 1024 * 1024) // pixel_count))
    with tempfile.TemporaryDirectory(prefix="dfl-absdiff-") as temporary:
        images = np.memmap(Path(temporary) / "images.bin", mode="w+", dtype=np.uint8,
                           shape=(len(image_paths), *shape))
        try:
            for index, filename in enumerate(io.progress_bar_generator(image_paths, "读取图片")):
                image = first if index == 0 else cv2_imread(filename)
                if image is None or image.shape != shape:
                    raise ValueError(f"absdiff 要求图片可读且尺寸一致：{filename}")
                images[index] = image
            images.flush()
            visited = np.zeros(len(image_paths), dtype=bool)
            current = 0
            ordered = [current]
            visited[current] = True
            for _ in io.progress_bar_generator(range(len(image_paths) - 1), "排序中"):
                candidates = np.flatnonzero(~visited)
                distances = np.empty(len(candidates), dtype=np.int64)
                reference = images[current].astype(np.int16)
                for offset in range(0, len(candidates), batch_size):
                    selected = candidates[offset:offset + batch_size]
                    differences = images[selected].astype(np.int16) - reference
                    distances[offset:offset + len(selected)] = np.abs(differences).sum(axis=(1, 2, 3), dtype=np.int64)
                selected_index = np.argmin(distances) if is_sim else np.argmax(distances)
                current = int(candidates[selected_index])
                visited[current] = True
                ordered.append(current)
        finally:
            images._mmap.close()
    return [(image_paths[index],) for index in ordered], []

def final_process(input_path, img_list, trash_img_list, *, dry_run=False, sort_method=None, preserved_files=(), quality_context=None):
    input_path = Path(input_path).absolute()
    changes = []
    for index, row in enumerate(img_list):
        source = Path(row[0]).absolute()
        if source.parent != input_path:
            raise ValueError('排序源文件必须直接位于当前 aligned 目录')
        target = f'{quality_context["targetPrefix"]}{index:05d}{source.suffix}' if quality_context else f'{index:05d}{source.suffix}'
        changes.append({'source': source.name, 'target': target, 'location': 'input'})
        sidecar = source.with_suffix(source.suffix + '.landmarks.json')
        if sidecar.exists():
            changes.append({'source': sidecar.name, 'target': target + '.landmarks.json', 'location': 'input'})
    for row in trash_img_list:
        source = Path(row[0]).absolute()
        if source.parent != input_path:
            raise ValueError('排序源文件必须直接位于当前 aligned 目录')
        changes.append({'source': source.name, 'target': source.name, 'location': 'discarded'})
        sidecar = source.with_suffix(source.suffix + '.landmarks.json')
        if sidecar.exists():
            changes.append({'source': sidecar.name, 'target': sidecar.name, 'location': 'discarded'})
    # Back up the complete faceset, preserving the exact original names and
    # bytes outside the selected quality window. No hand-moving is needed.
    for filename in preserved_files:
        source = Path(filename).absolute()
        if source.parent != input_path:
            raise ValueError('保留源文件必须直接位于当前 aligned 目录')
        changes.append({'source': source.name, 'target': source.name, 'location': 'input'})
        sidecar = source.with_suffix(source.suffix + '.landmarks.json')
        if sidecar.exists():
            changes.append({'source': sidecar.name, 'target': sidecar.name, 'location': 'input'})
    scores = []
    for kept, rows in ((True, img_list), (False, trash_img_list)):
        for row in rows:
            if quality_context and len(row) >= 6 and isinstance(row[5], dict):
                if file_sha256(row[0]) != row[5]['sourceSha256']:
                    raise ValueError(f'评分后源图发生变化：{Path(row[0]).name}')
                scores.append({'source': Path(row[0]).name, 'selected': kept, 'value': float(row[1]), **row[5]})
            elif len(row) >= 6 and isinstance(row[5], dict):
                scores.append({'source': Path(row[0]).name, 'selected': kept,
                               'metric': row[5]['metric'], 'value': float(row[1]),
                               'yaw': float(row[3]), 'pitch': float(row[4]),
                               'source_box_available': row[5]['source_box_available']})
    # Keep the receipt readable even for unusually large legacy CLI batches.
    details = {'sort_method': sort_method, 'selected_count': len(img_list),
               'archived_count': len(trash_img_list), 'scores': scores[:2000],
               'scores_truncated': len(scores) > 2000}
    if quality_context:
        details.update({'qualityCoverage': quality_context, 'scores': scores,
                        'scores_truncated': False, 'preserved_outside_count': len(preserved_files)})
    receipt = execute_plan(input_path, changes, operation='sort', dry_run=dry_run, details=details)
    io.log_info(f'排序预览：保留 {len(img_list)}，归档 {len(trash_img_list)}。' if dry_run else f'排序批次已提交；原件和恢复回执：{receipt["receipt_path"]}')
    return receipt

sort_func_methods = {
    'blur':        ("blur", sort_by_blur),
    'motion-blur': ("motion_blur", sort_by_motion_blur),
    'face-yaw':    ("人脸偏航（yaw）方向", sort_by_face_yaw),
    'face-pitch':  ("人脸俯仰（pitch）方向", sort_by_face_pitch),
    'face-source-rect-size' : ("源图中人脸框大小", sort_by_face_source_rect_size),
    'hist':        ("直方图相似度", sort_by_hist),
    'hist-dissim': ("直方图不相似度", sort_by_hist_dissim),
    'brightness':  ("亮度", sort_by_brightness),
    'hue':         ("色相", sort_by_hue),
    'black':       ("黑色像素占比", sort_by_black),
    'origname':    ("原始文件名", sort_by_origname),
    'oneface':     ("仅保留单人脸图片", sort_by_oneface_in_image),
    'absdiff':     ("绝对像素差", sort_by_absdiff),
    'final':       ("姿态覆盖筛选（CPBD 清晰度）", sort_best),
    'final-fast':  ("姿态覆盖筛选（源框面积）", sort_best_faster),
    'final-by-blur': ("姿态覆盖筛选（CPBD 清晰度）", sort_best),
    'final-by-size': ("姿态覆盖筛选（源框面积）", sort_best_faster),
    'quality-coverage': ("前景细节代理 + 姿态/眼口几何覆盖（每轮 500）", sort_quality_coverage),
}

def main (input_path, sort_by_method=None, *, dry_run=False, target_count=None, offset=0, limit=500, quality_metric=None):
    io.log_info ("正在运行排序工具（Sorter）。\r\n")

    if sort_by_method is None:
        io.log_info(f"请选择排序方式：")

        key_list = list(sort_func_methods.keys())
        for i, key in enumerate(key_list):
            desc, func = sort_func_methods[key]
            io.log_info(f"[{i}] {desc}")

        io.log_info("")
        id = io.input_int("", 5, valid_list=[*range(len(key_list))] )

        sort_by_method = key_list[id]
    else:
        sort_by_method = sort_by_method.lower()

    desc, func = sort_func_methods[sort_by_method]
    if func is sort_quality_coverage:
        img_list, trash_img_list, preserved, context = func(input_path, target_count=target_count,
            offset=offset, limit=limit, quality_metric=quality_metric)
        return final_process(input_path, img_list, trash_img_list, dry_run=dry_run,
            sort_method=sort_by_method, preserved_files=preserved, quality_context=context)
    img_list, trash_img_list = func(input_path, target_count=target_count) if func in (sort_best, sort_best_faster) else func(input_path)

    return final_process(input_path, img_list, trash_img_list, dry_run=dry_run, sort_method=sort_by_method)
