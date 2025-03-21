# Copyright (c) Ye Liu. Licensed under the BSD 3-Clause License.

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.wrappers import GroundingDataset
from videomind.utils.parser import parse_query


@DATASETS.register(name='vid_morp')
class VidMorpDataset(GroundingDataset):

    ANNO_PATH = 'data/vid_morp/anno_vid_morp.jsonl'

    VIDEO_ROOT = 'data/vid_morp/videos_3fps_480_noaudio'
    DURATIONS = 'data/vid_morp/durations.json'

    UNIT = 0.001

    @classmethod
    def load_annos(self, split='train'):
        assert split == 'train'

        raw_annos = nncore.load(self.ANNO_PATH)
        durations = nncore.load(self.DURATIONS)

        annos = []
        for raw_anno in raw_annos:
            vid = raw_anno['vid']

            if vid not in durations:
                continue

            anno = dict(
                source='vid_morp',
                data_type='grounding',
                video_path=nncore.join(self.VIDEO_ROOT, vid + '.mp4'),
                duration=durations[vid],
                query=parse_query(raw_anno['query']),
                span=[raw_anno['span']])

            annos.append(anno)

        return annos
