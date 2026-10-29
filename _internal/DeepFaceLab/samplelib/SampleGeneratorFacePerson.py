import copy
import multiprocessing
import traceback

import cv2
import numpy as np

from core import mplib
from core.joblib import SubprocessGenerator, ThisThreadGenerator
from facelib import LandmarksProcessor
from samplelib import (SampleGeneratorBase, SampleLoader, SampleProcessor,
                       SampleType)



class Index2DHost():
    """
    Provides random shuffled 2D indexes for multiprocesses
    """
    def __init__(self, indexes2D):
        self.sq = multiprocessing.Queue()
        self.cqs = []
        self.clis = []
        self.thread = threading.Thread(target=self.host_thread, args=(indexes2D,) )
        self.thread.daemon = True
        self.thread.start()

    def host_thread(self, indexes2D):
        indexes_counts_len = len(indexes2D)

        idxs = [*range(indexes_counts_len)]
        idxs_2D = [None]*indexes_counts_len
        shuffle_idxs = []
        shuffle_idxs_2D = [None]*indexes_counts_len
        for i in range(indexes_counts_len):
            idxs_2D[i] = indexes2D[i]
            shuffle_idxs_2D[i] = []

        sq = self.sq

        while True:
            while not sq.empty():
                obj = sq.get()
                cq_id, cmd = obj[0], obj[1]

                if cmd == 0: #get_1D
                    count = obj[2]

                    result = []
                    for i in range(count):
                        if len(shuffle_idxs) == 0:
                            shuffle_idxs = idxs.copy()
                            np.random.shuffle(shuffle_idxs)
                        result.append(shuffle_idxs.pop())
                    self.cqs[cq_id].put (result)
                elif cmd == 1: #get_2D
                    targ_idxs,count = obj[2], obj[3]
                    result = []

                    for targ_idx in targ_idxs:
                        sub_idxs = []
                        for i in range(count):
                            ar = shuffle_idxs_2D[targ_idx]
                            if len(ar) == 0:
                                ar = shuffle_idxs_2D[targ_idx] = idxs_2D[targ_idx].copy()
                                np.random.shuffle(ar)
                            sub_idxs.append(ar.pop())
                        result.append (sub_idxs)
                    self.cqs[cq_id].put (result)

            time.sleep(0.001)

    def create_cli(self):
        cq = multiprocessing.Queue()
        self.cqs.append ( cq )
        cq_id = len(self.cqs)-1
        return Index2DHost.Cli(self.sq, cq, cq_id)

    # disable pickling
    def __getstate__(self):
        return dict()
    def __setstate__(self, d):
        self.__dict__.update(d)

    class Cli():
        def __init__(self, sq, cq, cq_id):
            self.sq = sq
            self.cq = cq
            self.cq_id = cq_id

        def get_1D(self, count):
            self.sq.put ( (self.cq_id,0, count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

        def get_2D(self, idxs, count):
            self.sq.put ( (self.cq_id,1,idxs,count) )

            while True:
                if not self.cq.empty():
                    return self.cq.get()
                time.sleep(0.001)

'''
arg
output_sample_types = [
                        [SampleProcessor.TypeFlags, size, (optional) {} opts ] ,
                        ...
                      ]
'''
class SampleGeneratorFacePerson(SampleGeneratorBase):
    def __init__ (self, samples_path, debug=False, batch_size=1,
                        sample_process_options=SampleProcessor.Options(),
                        output_sample_types=[],
                        person_id_mode=1,
                        **kwargs):

        super().__init__(debug, batch_size)
        self.sample_process_options = sample_process_options
        self.output_sample_types = output_sample_types
        self.person_id_mode = person_id_mode

        raise NotImplementedError("Currently SampleGeneratorFacePerson is not implemented.")

        samples_host = SampleLoader.mp_host (SampleType.FACE, samples_path)
        samples = samples_host.get_list()
        self.samples_len = len(samples)

        if self.samples_len == 0:
            raise ValueError('No training data provided.')

        unique_person_names = { sample.person_name for sample in samples }
        persons_name_idxs = { person_name : [] for person_name in unique_person_names }
        for i,sample in enumerate(samples):
            persons_name_idxs[sample.person_name].append (i)
        indexes2D = [ persons_name_idxs[person_name] for person_name in unique_person_names ]
        index2d_host = Index2DHost(indexes2D)

        if self.debug:
            self.generators_count = 1
            self.generators = [iter_utils.ThisThreadGenerator ( self.batch_func, (samples_host.create_cli(), index2d_host.create_cli(),) )]
        else:
            self.generators_count = np.clip(multiprocessing.cpu_count(), 2, 4)
            self.generators = [iter_utils.SubprocessGenerator ( self.batch_func, (samples_host.create_cli(), index2d_host.create_cli(),) ) for i in range(self.generators_count) ]

        self.generator_counter = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.generator_counter += 1
        generator = self.generators[self.generator_counter % len(self.generators) ]
        return next(generator)

    def batch_func(self, param ):
        samples, index2d_host, = param
        bs = self.batch_size

        while True:
            person_idxs = index2d_host.get_1D(bs)
            samples_idxs = index2d_host.get_2D(person_idxs, 1)

            batches = None
            for n_batch in range(bs):
                person_id = person_idxs[n_batch]
                sample_idx = samples_idxs[n_batch][0]

                sample = samples[ sample_idx ]
                try:
                    x, = SampleProcessor.process ([sample], self.sample_process_options, self.output_sample_types, self.debug)
                except:
                    raise Exception ("Exception occured in sample %s. Error: %s" % (sample.filename, traceback.format_exc() ) )

                if batches is None:
                    batches = [ [] for _ in range(len(x)) ]

                    batches += [ [] ]
                    i_person_id = len(batches)-1

                for i in range(len(x)):
                    batches[i].append ( x[i] )

                batches[i_person_id].append ( np.array([person_id]) )

            yield [ np.array(batch) for batch in batches]

    @staticmethod
    def get_person_id_max_count(samples_path):
        return SampleLoader.get_person_id_max_count(samples_path)
